"""Load event.yaml and resolve project paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return ROOT


def load_event(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (ROOT / "event.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid event config: {cfg_path}")
    return cfg


def download_bbox(cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return (min_lon, max_lon, min_lat, max_lat) including download margin."""
    bbox = list(cfg["domain"]["bbox"])
    margin = float(cfg["domain"].get("download_margin_deg", 0.5))
    return (
        bbox[0] - margin,
        bbox[1] + margin,
        bbox[2] - margin,
        bbox[3] + margin,
    )


def plot_bbox(cfg: dict[str, Any]) -> list[float]:
    return list(cfg["domain"]["bbox"])


def ensure_dirs(cfg: dict[str, Any]) -> None:
    root = project_root()
    for key in ("data_dir", "figures_dir", "spatial_dir", "point_dir"):
        (root / cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    for src in ("gfs", "hycom", "era5", "cmems"):
        (root / cfg["paths"]["data_dir"] / src).mkdir(parents=True, exist_ok=True)
        (root / cfg["paths"]["spatial_dir"] / src).mkdir(parents=True, exist_ok=True)
