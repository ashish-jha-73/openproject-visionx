from __future__ import annotations
import numpy as np
import torch
import json
import random
from pathlib import Path
from typing import Any

from reconstruction3d.config import AppConfig, DatasetConfig, ProjectConfig, TrainingConfig


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_config(path: str | Path | None) -> ProjectConfig:
    if path is None:
        return ProjectConfig()

    payload = json.loads(Path(path).read_text())
    dataset = DatasetConfig(**payload.get("dataset", {}))
    training = TrainingConfig(**payload.get("training", {}))
    app = AppConfig(**payload.get("app", {}))
    return ProjectConfig(dataset=dataset, training=training, app=app)


def save_checkpoint(path: str | Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, target)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location: str = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint.get("metadata", {})
