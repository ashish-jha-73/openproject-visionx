from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from reconstruction3d.config import DatasetConfig


def default_image_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def voxel_iou(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred_binary = (prediction >= threshold).float()
    target_binary = (target >= threshold).float()
    intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3, 4))
    union = ((pred_binary + target_binary) > 0).float().sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    return intersection / union


def normalize_points(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.abs(centered).max()
    if scale < 1e-6:
        return centered
    return centered / scale


def voxelize_points(points: np.ndarray, voxel_size: int) -> np.ndarray:
    normalized = normalize_points(points)
    scaled = ((normalized + 1.0) * 0.5) * (voxel_size - 1)
    indices = np.clip(np.rint(scaled), 0, voxel_size - 1).astype(np.int64)
    voxels = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    return voxels


class SyntheticMultiViewDataset(Dataset):
    """Generates simple voxel primitives and projection-based views for smoke tests."""

    def __init__(self, config: DatasetConfig, split: str = "train") -> None:
        self.config = config
        self.split = split
        self.length = config.synthetic_samples if split == "train" else max(32, config.synthetic_samples // 4)
        self.image_transform = default_image_transform(config.image_size)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(seed=index + (0 if self.split == "train" else 10_000))
        volume = self._random_volume(rng)
        views = self._render_views(volume)
        return {
            "images": views,
            "voxels": torch.from_numpy(volume[None, ...].astype(np.float32)),
        }

    def _random_volume(self, rng: np.random.Generator) -> np.ndarray:
        size = self.config.voxel_size
        grid = np.indices((size, size, size)).astype(np.float32)
        center = rng.uniform(low=size * 0.3, high=size * 0.7, size=(3, 1, 1, 1)).astype(np.float32)
        radii = rng.uniform(low=size * 0.12, high=size * 0.22, size=(3, 1, 1, 1)).astype(np.float32)
        normalized = ((grid - center) / radii) ** 2
        ellipsoid = normalized.sum(axis=0) <= 1.0

        if rng.random() > 0.5:
            start = rng.integers(low=size // 5, high=size // 2, size=3)
            box_extent = rng.integers(low=size // 6, high=size // 3, size=3)
            cube = np.zeros_like(ellipsoid)
            cube[
                start[0] : start[0] + box_extent[0],
                start[1] : start[1] + box_extent[1],
                start[2] : start[2] + box_extent[2],
            ] = True
            ellipsoid = np.logical_or(ellipsoid, cube)

        return ellipsoid.astype(np.float32)

    def _render_views(self, volume: np.ndarray) -> torch.Tensor:
        views = []
        num_views = self.config.num_views
        for view_index in range(num_views):
            rotated = np.rot90(volume, k=view_index % 4, axes=(0, 1))
            if view_index % 2 == 1:
                rotated = np.rot90(rotated, k=1, axes=(1, 2))
            projection = rotated.max(axis=2)
            depth_hint = rotated.mean(axis=2)
            image = np.stack([projection, depth_hint, projection * 0.7 + depth_hint * 0.3], axis=-1)
            pil = Image.fromarray((image * 255).astype(np.uint8))
            views.append(self.image_transform(pil))
        return torch.stack(views, dim=0)


class ShapeNetMultiViewDataset(Dataset):
    """
    Expected layout:
    data/shapenet/
      train.json
      val.json

    Each JSON file should contain:
    [
      {
        "images": ["relative/path/view_0.png", ...],
        "voxels": "relative/path/model.npy"
      }
    ]
    """

    def __init__(self, config: DatasetConfig, split: str = "train") -> None:
        self.config = config
        self.root = Path(config.dataset_root)
        self.split = split
        manifest_name = f"{split}.json"
        manifest_path = self.root / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"ShapeNet manifest not found at {manifest_path}. "
                "Create train.json/val.json or use dataset_type='synthetic'."
            )

        self.records = json.loads(manifest_path.read_text())
        self.image_transform = default_image_transform(config.image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        image_paths = record["images"][: self.config.num_views]
        images = []
        for image_path in image_paths:
            image = Image.open(self.root / image_path).convert("RGB")
            images.append(self.image_transform(image))

        if len(images) < self.config.num_views:
            raise ValueError(f"Record {index} only has {len(images)} views; expected {self.config.num_views}.")

        voxel_path = self.root / record["voxels"]
        voxels = np.load(voxel_path).astype(np.float32)
        if voxels.shape != (self.config.voxel_size, self.config.voxel_size, self.config.voxel_size):
            raise ValueError(
                f"Unexpected voxel shape {voxels.shape}. "
                f"Expected {(self.config.voxel_size, self.config.voxel_size, self.config.voxel_size)}."
            )

        return {
            "images": torch.stack(images, dim=0),
            "voxels": torch.from_numpy(voxels[None, ...]),
        }


class ShapeNetPointCloudDataset(Dataset):
    """
    Supports ShapeNet-style point cloud folders with split files such as:
    shapenet/train_test_split/shuffled_train_file_list.json

    Each sample is loaded from either:
    - <synset>/<shape_id>_8x8.npz
    - <synset>/<shape_id>.txt
    """

    def __init__(self, config: DatasetConfig, split: str = "train") -> None:
        self.config = config
        self.root = Path(config.dataset_root)
        self.split = split
        split_map = {
            "train": "shuffled_train_file_list.json",
            "val": "shuffled_val_file_list.json",
            "test": "shuffled_test_file_list.json",
        }
        split_filename = split_map.get(split, f"shuffled_{split}_file_list.json")
        split_path = self.root / "train_test_split" / split_filename
        if not split_path.exists():
            raise FileNotFoundError(
                f"Point-cloud split file not found at {split_path}. "
                "Expected shuffled_train/val/test_file_list.json inside train_test_split/."
            )

        entries = json.loads(split_path.read_text())
        filtered_entries = self._filter_entries(entries)
        self.records = [self._resolve_entry(entry) for entry in filtered_entries]

    def _filter_entries(self, entries: list[str]) -> list[str]:
        if self.config.category_filter:
            allowed = set(self.config.category_filter)
            entries = [entry for entry in entries if Path(entry).parts[1] in allowed]

        split_limits = {
            "train": self.config.max_train_samples,
            "val": self.config.max_val_samples,
            "test": self.config.max_test_samples,
        }
        limit = split_limits.get(self.split)
        if limit is not None:
            entries = entries[:limit]

        if not entries:
            raise ValueError(
                f"No records left for split='{self.split}'. "
                "Check category_filter and max_*_samples in your config."
            )
        return entries

    def _resolve_entry(self, entry: str) -> dict[str, Path]:
        _, synset_id, shape_id = Path(entry).parts
        npz_path = self.root / synset_id / f"{shape_id}_8x8.npz"
        txt_path = self.root / synset_id / f"{shape_id}.txt"
        if npz_path.exists():
            return {"synset_id": synset_id, "shape_id": shape_id, "source": npz_path, "kind": "npz"}
        if txt_path.exists():
            return {"synset_id": synset_id, "shape_id": shape_id, "source": txt_path, "kind": "txt"}
        raise FileNotFoundError(f"Could not find either {npz_path.name} or {txt_path.name} for {entry}.")

    def __len__(self) -> int:
        return len(self.records)

    def _load_arrays(self, record: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
        if record["kind"] == "npz":
            payload = np.load(record["source"])
            points = payload["pc"].astype(np.float32)
            normals = payload["sn"].astype(np.float32)
            return points, normals

        raw = np.loadtxt(record["source"], dtype=np.float32)
        points = raw[:, :3]
        normals = raw[:, 3:6]
        return points, normals

    def _sample_points(self, points: np.ndarray, normals: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed=index)
        replace = len(points) < self.config.num_points
        choice = rng.choice(len(points), size=self.config.num_points, replace=replace)
        return points[choice], normals[choice]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        points, normals = self._load_arrays(record)
        points, normals = self._sample_points(points, normals, index)
        normalized_points = normalize_points(points)
        voxels = voxelize_points(normalized_points, self.config.voxel_size)

        if self.config.use_normals:
            features = np.concatenate([normalized_points, normals], axis=1)
        else:
            features = normalized_points

        return {
            "points": torch.from_numpy(features.astype(np.float32)),
            "voxels": torch.from_numpy(voxels[None, ...]),
        }


def build_dataset(config: DatasetConfig, split: str) -> Dataset:
    if config.dataset_type == "synthetic":
        return SyntheticMultiViewDataset(config, split)
    if config.dataset_type == "shapenet":
        return ShapeNetMultiViewDataset(config, split)
    if config.dataset_type == "shapenet_pointcloud":
        return ShapeNetPointCloudDataset(config, split)
    raise ValueError(f"Unsupported dataset_type: {config.dataset_type}")
