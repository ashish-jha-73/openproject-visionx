# 3D Object Reconstruction From Multiple Views

This project reconstructs a 3D voxel volume from several RGB images of the same object taken from different viewpoints.

It includes:

- a multi-view encoder-decoder reconstruction model in PyTorch
- a synthetic dataset for local smoke testing
- a ShapeNet-compatible dataset loader for real training data
- a training CLI for producing checkpoints
- an inference CLI for reconstructing a volume from uploaded images
- a Gradio demo app for interactive testing

## Project Structure

```text
src/reconstruction3d/
  app.py          # Gradio demo app
  config.py       # Typed config objects
  data.py         # Synthetic + ShapeNet dataset loaders
  infer.py        # Inference CLI
  model.py        # Multi-view reconstruction network
  train.py        # Training loop
  utils.py        # Shared utilities
config.example.json
config.pointcloud.json
```

## Source File Descriptions

### `src/reconstruction3d/__init__.py`
- Use: marks `reconstruction3d` as a Python package so the modules can be imported.
- What it contains: package-level export names in `__all__`.
- Main Python features used: basic module exports only.

### `src/reconstruction3d/config.py`
- Use: stores structured configuration for dataset, training, and app settings.
- Main classes/functions: `DatasetConfig`, `TrainingConfig`, `AppConfig`, `ProjectConfig`.
- Main libraries used: `dataclasses`, `pathlib`.

### `src/reconstruction3d/utils.py`
- Use: shared helper functions used across training, inference, and app code.
- Main functions: `set_seed()`, `ensure_dir()`, `load_config()`, `save_checkpoint()`, `load_checkpoint()`.
- Main libraries used: `json`, `random`, `numpy`, `torch`, `pathlib`.

### `src/reconstruction3d/data.py`
- Use: loads training data for the model.
- Main functions/classes: `default_image_transform()`, `voxel_iou()`, `SyntheticMultiViewDataset`, `ShapeNetMultiViewDataset`, `build_dataset()`.
- Main libraries used: `torch`, `torchvision.transforms`, `torch.utils.data.Dataset`, `numpy`, `PIL.Image`, `json`, `pathlib`.
- Why it matters: this file handles both quick local testing with synthetic data and real dataset loading for ShapeNet-style training.

### `src/reconstruction3d/model.py`
- Use: defines the neural network used for multi-view 3D reconstruction.
- Main classes: `ViewEncoder`, `VolumeDecoder`, `MultiViewReconstructionNet`.
- Main layers/functions used: `Conv2d`, `BatchNorm2d`, `AdaptiveAvgPool2d`, `Linear`, `ConvTranspose3d`, `Conv3d`, `ReLU`, `sigmoid`.
- Main libraries used: `torch`, `torch.nn`.

### `src/reconstruction3d/train.py`
- Use: trains the reconstruction model and saves checkpoints.
- Main functions: `parse_args()`, `run_epoch()`, `main()`.
- Main libraries used: `argparse`, `torch`, `torch.nn`, `torch.utils.data.DataLoader`, `tqdm`.
- What it does: builds datasets, creates dataloaders, runs training and validation loops, computes loss and IoU, and writes `best.pt` / `latest.pt`.

### `src/reconstruction3d/infer.py`
- Use: runs prediction on multiple input images and saves the reconstructed voxel output.
- Main functions: `parse_args()`, `build_transform()`, `load_images()`, `save_volume_preview()`, `main()`.
- Main libraries used: `argparse`, `torch`, `numpy`, `PIL.Image`, `torchvision.transforms`, `pathlib`.
- What it outputs: a `.npy` voxel volume file and a `.png` preview image.

### `src/reconstruction3d/app.py`
- Use: launches the Gradio web app for interactive testing in the browser.
- Main functions/classes: `parse_args()`, `build_transform()`, `ReconstructionApp`, `build_interface()`, `main()`.
- Main libraries used: `gradio`, `torch`, `numpy`, `PIL.Image`, `torchvision.transforms`, `argparse`, `pathlib`.
- What it does: loads the checkpoint, accepts uploaded images, runs reconstruction, and displays preview results and stats.

## Setup

1. Open terminal in this project folder:

```bash
cd "/Users/sushanthgunda/Desktop/cv project"
```

2. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -e .
```

4. Optional check that the package is installed:

```bash
python3 -c "import reconstruction3d; print('package loaded')"
```

## Train With Synthetic Data

This mode is useful for confirming that the pipeline works end-to-end without downloading ShapeNet.

Run this exact command:

```bash
reconstruct3d-train --config config.example.json --epochs 2
```

What this does:

- uses the synthetic dataset from the config
- trains the model for 2 epochs
- saves checkpoints in `checkpoints/`

Expected output files:

- `checkpoints/latest.pt`
- `checkpoints/best.pt`

## Train With ShapeNet

1. Edit `config.example.json` and change:

```json
"dataset_type": "shapenet"
```

2. Create this dataset structure:

```text
data/shapenet/
  train.json
  val.json
  renders/object_001_view_0.png
  renders/object_001_view_1.png
  voxels/object_001.npy
```

3. Create `train.json` and `val.json`.

Each manifest record should look like this:

```json
[
  {
    "images": [
      "renders/object_001_view_0.png",
      "renders/object_001_view_1.png",
      "renders/object_001_view_2.png",
      "renders/object_001_view_3.png"
    ],
    "voxels": "voxels/object_001.npy"
  }
]
```

The voxel target is expected to be a `(32, 32, 32)` NumPy array by default.

4. Run training:

```bash
reconstruct3d-train --config config.example.json --epochs 5
```

## Train With The Downloaded ShapeNet Point-Cloud Dataset

Your downloaded `shapenet/` folder matches a point-cloud dataset layout, not the `train.json` / `val.json` image-manifest format above.

Use this config instead:

```bash
reconstruct3d-train --config config.pointcloud.json --epochs 5
```

What this mode does:

- reads split files from `shapenet/train_test_split/`
- loads point clouds from `*_8x8.npz` or `.txt`
- samples a fixed number of points per object
- voxelizes the points on the fly into a `32 x 32 x 32` occupancy target
- trains a PointNet-style encoder with a 3D volume decoder

Fast training controls in `config.pointcloud.json`:

- `category_filter`: limit training to one or more ShapeNet category ids such as `["03001627"]` for chairs
- `max_train_samples`: use only the first N training samples
- `max_val_samples`: use only the first N validation samples
- `num_points`: reduce points per object for faster training

Example fast setup:

- `category_filter: ["03001627"]`
- `max_train_samples: 500`
- `max_val_samples: 100`
- `num_points: 512`
- `epochs: 2`

Expected output files:

- `checkpoints_pointcloud/latest.pt`
- `checkpoints_pointcloud/best.pt`

Important:

- this training mode is for your downloaded point-cloud dataset
- the Gradio app is still for the image-based demo flow
- `config.example.json` is still the right config for the app

## Run Inference

Make sure you already have a trained checkpoint such as `checkpoints/best.pt`.

Run inference with exact steps:

1. Put 4 images of the same object in one folder.
2. Run:

```bash
reconstruct3d-infer \
  --checkpoint checkpoints/best.pt \
  --output outputs/chair.npy \
  examples/view1.png examples/view2.png examples/view3.png examples/view4.png
```

If you want to test with the sample images already generated in this project, run:

```bash
reconstruct3d-infer \
  --checkpoint checkpoints/best.pt \
  --output outputs/smoke.npy \
  tmp_views/view_0.png tmp_views/view_1.png tmp_views/view_2.png tmp_views/view_3.png
```

This writes:

- the reconstructed volume to `outputs/chair.npy`
- a projection preview to `outputs/chair.png`

## Launch The App

Make sure you already have a checkpoint file such as `checkpoints/best.pt`.

Run the app with the installed CLI:

```bash
reconstruct3d-app --config config.example.json
```

Or run the Python file directly:

```bash
python3 -m reconstruction3d.app --config config.example.json
```

Then:

1. Wait for Gradio to print a local URL in the terminal.
2. Open that URL in your browser.
3. Upload the object images from multiple views.
4. Click `Reconstruct`.
5. Check the voxel preview and reconstruction stats.

## Full Run Order

If you want the exact sequence from start to finish, use this order:

1. Open terminal in the project:

```bash
cd "/Users/sushanthgunda/Desktop/cv project"
```

2. Create and activate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the project:

```bash
pip install -e .
```

4. Train a quick test model:

```bash
reconstruct3d-train --config config.example.json --epochs 2
```

5. Run inference:

```bash
reconstruct3d-infer \
  --checkpoint checkpoints/best.pt \
  --output outputs/smoke.npy \
  tmp_views/view_0.png tmp_views/view_1.png tmp_views/view_2.png tmp_views/view_3.png
```

6. Launch the browser demo:

```bash
reconstruct3d-app --config config.example.json
```

## Notes

- The included model is intentionally compact so the project is easy to understand and extend.
- For stronger results on ShapeNet, you will likely want camera-aware fusion, better supervision, and mesh extraction.
- The synthetic dataset exists for workflow validation, not benchmark quality.
- The downloaded `shapenet/` folder in this workspace is currently used through `config.pointcloud.json`, not through `train.json` / `val.json`.
