from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from torchvision import transforms

from reconstruction3d.config import ProjectConfig
from reconstruction3d.model import MultiViewReconstructionNet
from reconstruction3d.utils import load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the reconstruction demo app.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    return parser.parse_args()


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


class ReconstructionApp:
    def __init__(self, config: ProjectConfig) -> None:
        if config.dataset.dataset_type == "shapenet_pointcloud":
            raise ValueError(
                "The Gradio app currently supports image-based reconstruction demos only. "
                "Use config.example.json for the app and config.pointcloud.json for point-cloud training."
            )
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = build_transform(config.dataset.image_size)
        self.model = MultiViewReconstructionNet(voxel_size=config.dataset.voxel_size).to(self.device)
        checkpoint_path = Path(config.app.checkpoint_path)
        if checkpoint_path.exists():
            load_checkpoint(checkpoint_path, self.model, map_location=self.device.type)
        self.model.eval()
        self.sample_dir = Path("tmp_views")
        self.viewer_dir = Path("outputs/gradio_models")
        self.viewer_dir.mkdir(parents=True, exist_ok=True)

    def _select_occupied_voxels(self, volume: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, float]:
        occupied = np.argwhere(volume >= threshold)
        used_threshold = threshold

        if len(occupied) == 0:
            used_threshold = float(np.quantile(volume, 0.995))
            occupied = np.argwhere(volume >= used_threshold)

        if len(occupied) == 0:
            used_threshold = float(volume.max())
            occupied = np.argwhere(volume >= used_threshold)

        values = volume[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
        return occupied, values, used_threshold

    def _write_open3d_mesh(self, volume: np.ndarray, threshold: float) -> tuple[str, int, float]:
        occupied, values, used_threshold = self._select_occupied_voxels(volume, threshold)
        center = (np.array(volume.shape, dtype=np.float32) - 1.0) / 2.0
        scale = float(max(volume.shape) / 2.0)
        normalized_points = (occupied.astype(np.float32) - center) / scale
        max_value = max(float(values.max(initial=1.0)), 1e-6)
        normalized_values = values / max_value

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(normalized_points)
        colors = np.stack(
            [
                np.clip(normalized_values, 0.0, 1.0),
                np.clip(0.85 - 0.25 * normalized_values, 0.0, 1.0),
                np.ones_like(normalized_values),
            ],
            axis=1,
        )
        point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        mesh_path = self.viewer_dir / f"reconstruction_{uuid4().hex}.ply"
        if len(occupied) >= 4:
            if len(normalized_points) > 600:
                strongest = np.argsort(normalized_values)[-600:]
                mesh_points = normalized_points[strongest]
            else:
                mesh_points = normalized_points

            mesh_cloud = o3d.geometry.PointCloud()
            mesh_cloud.points = o3d.utility.Vector3dVector(mesh_points)
            mesh_cloud.estimate_normals()
            previous_verbosity = o3d.utility.get_verbosity_level()
            try:
                o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(mesh_cloud, 0.12)
            finally:
                o3d.utility.set_verbosity_level(previous_verbosity)
            if len(mesh.triangles) > 0:
                mesh.compute_vertex_normals()
                o3d.io.write_triangle_mesh(str(mesh_path), mesh)
                return str(mesh_path), int(len(occupied)), used_threshold

        o3d.io.write_point_cloud(str(mesh_path), point_cloud)
        return str(mesh_path), int(len(occupied)), used_threshold

    def _build_projection_panel(self, volume: np.ndarray, threshold: float) -> Image.Image:
        occupied = (volume >= threshold).astype(np.float32)
        if occupied.sum() == 0:
            occupied = (volume >= np.quantile(volume, 0.995)).astype(np.float32)

        def to_panel(array: np.ndarray) -> np.ndarray:
            if array.max() > 0:
                normalized = array / array.max()
            else:
                normalized = array
            image = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
            rgb = np.stack([image, np.clip(image * 0.8 + 30, 0, 255).astype(np.uint8), 255 - image // 3], axis=-1)
            return np.kron(rgb, np.ones((6, 6, 1), dtype=np.uint8))

        xy = to_panel(occupied.max(axis=2))
        xz = to_panel(occupied.max(axis=1))
        yz = to_panel(occupied.max(axis=0))
        separator = np.full((xy.shape[0], 12, 3), 18, dtype=np.uint8)
        combined = np.concatenate([xy, separator, xz, separator, yz], axis=1)
        return Image.fromarray(combined)

    def predict(self, *inputs: Image.Image | float | None) -> tuple[str, Image.Image, str]:
        if not inputs:
            raise gr.Error("Upload at least one image to reconstruct an object.")

        *images, threshold_value = inputs
        threshold = float(threshold_value)
        valid_images = [image for image in images if image is not None]
        if not valid_images:
            raise gr.Error("Upload at least one image to reconstruct an object.")

        if len(valid_images) < self.config.dataset.num_views:
            while len(valid_images) < self.config.dataset.num_views:
                valid_images.append(valid_images[-1])

        tensor = torch.stack(
            [self.transform(image.convert("RGB")) for image in valid_images[: self.config.dataset.num_views]],
            dim=0,
        ).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            volume = self.model(tensor).squeeze().cpu().numpy()

        model_path, occupied_count, used_threshold = self._write_open3d_mesh(volume, threshold)
        preview_image = self._build_projection_panel(volume, used_threshold)
        stats = (
            f"Resolution: {volume.shape[0]}^3 voxels\n"
            f"Mean occupancy: {volume.mean():.4f}\n"
            f"Displayed threshold: {used_threshold:.4f}\n"
            f"Displayed voxels: {occupied_count}\n"
            f"Occupied voxels (>0.5): {(volume > 0.5).sum()}"
        )
        return model_path, preview_image, stats

    def load_sample_images(self) -> tuple[Image.Image, ...]:
        sample_images: list[Image.Image] = []
        for index in range(self.config.dataset.num_views):
            image_path = self.sample_dir / f"view_{index}.png"
            if not image_path.exists():
                raise gr.Error(
                    f"Sample image not found: {image_path}. "
                    "Generate sample images first or upload your own files."
                )
            sample_images.append(Image.open(image_path).convert("RGB"))
        return tuple(sample_images)


def build_interface(config: ProjectConfig) -> gr.Blocks:
    app = ReconstructionApp(config)
    with gr.Blocks(title="Multi-view 3D Reconstruction") as interface:
        gr.Markdown(
            """
            # Multi-view 3D Reconstruction
            Upload images of the same object from several angles. The app predicts a voxel volume
            and shows both a 3D point-cloud view and a top-down occupancy preview.
            """
        )
        with gr.Row():
            image_inputs = [gr.Image(type="pil", label=f"View {index + 1}") for index in range(config.dataset.num_views)]
        threshold = gr.Slider(
            minimum=0.01,
            maximum=0.5,
            value=0.1,
            step=0.01,
            label="Visualization Threshold",
            info="Lower values show more predicted points in the 3D viewer.",
        )
        with gr.Row():
            load_samples = gr.Button("Load Sample Images")
            submit = gr.Button("Reconstruct")
        output_model = gr.Model3D(label="3D Reconstruction", display_mode="solid", clear_color=(0, 0, 0, 0))
        output_image = gr.Image(type="pil", label="Voxel Preview")
        output_stats = gr.Textbox(label="Reconstruction Stats")
        load_samples.click(fn=app.load_sample_images, outputs=image_inputs)
        submit.click(fn=app.predict, inputs=[*image_inputs, threshold], outputs=[output_model, output_image, output_stats])
    return interface


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    interface = build_interface(config)
    interface.launch(server_name=config.app.host, server_port=config.app.port)


if __name__ == "__main__":
    main()
