from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import argparse
from pathlib import Path

from reconstruction3d.model import MultiViewReconstructionNet
from reconstruction3d.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3D reconstruction inference.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output", type=str, default="outputs/reconstruction.npy", help="Output .npy path.")
    parser.add_argument("--image-size", type=int, default=64, help="Image size used by the model.")
    parser.add_argument("--voxel-size", type=int, default=32, help="Voxel resolution of the model.")
    parser.add_argument("images", nargs="+", help="Input images from different viewpoints.")
    return parser.parse_args()


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def load_images(image_paths: list[str], image_size: int) -> torch.Tensor:
    transform = build_transform(image_size)
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        images.append(transform(image))
    return torch.stack(images, dim=0).unsqueeze(0)


def save_volume_preview(volume: np.ndarray, output_path: Path) -> None:
    preview = volume.max(axis=0)
    image = Image.fromarray((preview * 255).astype(np.uint8))
    image.save(output_path.with_suffix(".png"))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiViewReconstructionNet(voxel_size=args.voxel_size).to(device)
    metadata = load_checkpoint(args.checkpoint, model, map_location=device.type)
    if metadata.get("input_mode") == "pointcloud":
        raise ValueError(
            "This checkpoint was trained from point clouds. "
            "The current inference CLI only supports image-based checkpoints."
        )
    if metadata.get("voxel_size") and metadata["voxel_size"] != args.voxel_size:
        raise ValueError(
            f"Checkpoint expects voxel_size={metadata['voxel_size']}, received {args.voxel_size}."
        )

    images = load_images(args.images, args.image_size).to(device)
    with torch.no_grad():
        volume = model(images).squeeze().cpu().numpy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, volume)
    save_volume_preview(volume, output_path)
    print(f"Saved reconstruction volume to {output_path}")
    print(f"Saved top-down preview to {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
