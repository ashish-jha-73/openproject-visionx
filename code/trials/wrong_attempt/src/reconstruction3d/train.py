from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from reconstruction3d.data import build_dataset, voxel_iou
from reconstruction3d.model import MultiViewReconstructionNet, PointCloudReconstructionNet
from reconstruction3d.utils import ensure_dir, load_config, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-view 3D reconstruction model.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    input_key: str,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, leave=False):
        inputs = batch[input_key].to(device)
        voxels = batch["voxels"].to(device)

        with torch.set_grad_enabled(is_training):
            prediction = model(inputs)
            loss = criterion(prediction, voxels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_iou += voxel_iou(prediction.detach(), voxels).mean().item()
        num_batches += 1

    return total_loss / max(1, num_batches), total_iou / max(1, num_batches)


def build_model(dataset_type: str, use_normals: bool, voxel_size: int) -> tuple[nn.Module, str, str]:
    if dataset_type in {"synthetic", "shapenet"}:
        return MultiViewReconstructionNet(voxel_size=voxel_size), "images", "multiview"
    if dataset_type == "shapenet_pointcloud":
        input_dim = 6 if use_normals else 3
        return PointCloudReconstructionNet(input_dim=input_dim, voxel_size=voxel_size), "points", "pointcloud"
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs

    set_seed()
    device = torch.device(config.training.device)
    train_dataset = build_dataset(config.dataset, split=config.dataset.train_split)
    val_dataset = build_dataset(config.dataset, split=config.dataset.val_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    model, input_key, input_mode = build_model(
        config.dataset.dataset_type,
        config.dataset.use_normals,
        config.dataset.voxel_size,
    )
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    checkpoint_dir = ensure_dir(config.training.checkpoint_dir)
    best_iou = -1.0

    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_iou = run_epoch(model, train_loader, criterion, optimizer, device, input_key)
        val_loss, val_iou = run_epoch(model, val_loader, criterion, None, device, input_key)
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(
            latest_path,
            model,
            metadata={
                "epoch": epoch,
                "val_iou": val_iou,
                "voxel_size": config.dataset.voxel_size,
                "dataset_type": config.dataset.dataset_type,
                "input_mode": input_mode,
            },
        )

        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                metadata={
                    "epoch": epoch,
                    "val_iou": val_iou,
                    "voxel_size": config.dataset.voxel_size,
                    "dataset_type": config.dataset.dataset_type,
                    "input_mode": input_mode,
                },
            )

    print(f"Training complete. Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
