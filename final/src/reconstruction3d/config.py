from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DatasetConfig:
    dataset_type: str = "synthetic"
    dataset_root: str = "data/shapenet"
    train_split: str = "train"
    val_split: str = "val"
    category_filter: list[str] = field(default_factory=list)
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    num_views: int = 4
    image_size: int = 64
    voxel_size: int = 32
    synthetic_samples: int = 256
    num_points: int = 2048
    use_normals: bool = True


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"
    num_workers: int = 0
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10


@dataclass(slots=True)
class AppConfig:
    checkpoint_path: str = "checkpoints/best.pt"
    host: str = "127.0.0.1"
    port: int = 7860


@dataclass(slots=True)
class ProjectConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    app: AppConfig = field(default_factory=AppConfig)

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.training.checkpoint_dir)
