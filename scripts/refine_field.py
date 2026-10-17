#!/usr/bin/env python3
"""Refine native fields onto event.yaml target_deg grid and write densified NetCDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import ensure_dirs, load_event, plot_bbox, project_root  # noqa: E402
from lib.io_util import save_dataset  # noqa: E402
from lib.refine import refine_dataset  # noqa: E402


INPUTS = {
    "gfs": [
        ("gfs_wave_subset.nc", "gfs_wave_refined.nc"),
        ("gfs_atmos_wind_subset.nc", "gfs_atmos_wind_refined.nc"),
    ],
    "hycom": [
        ("hycom_current_subset.nc", "hycom_current_refined.nc"),
    ],
    "era5": [
        ("era5_subset.nc", "era5_refined.nc"),
    ],
    "cmems": [
        ("cmems_current_subset.nc", "cmems_current_refined.nc"),
    ],
}


def refine_one(src: Path, dst: Path, cfg: dict) -> Path | None:
    if not src.exists():
        print(f"[SKIP] missing {src}")
        return None
    ds = xr.open_dataset(src)
    refine_cfg = cfg["refine"]
    out = refine_dataset(
        ds,
        target_deg=float(refine_cfg["target_deg"]),
        method=refine_cfg.get("method", "linear"),
        apply_gaussian=bool(refine_cfg.get("apply_gaussian", True)),
        gaussian_sigma=float(refine_cfg.get("gaussian_sigma", 0.8)),
        bbox=plot_bbox(cfg),
    )
    save_dataset(out, dst)
    ds.close()
    print(f"[OK] {dst.name} dims={dict(out.sizes)}")
    return dst


def run(event_path: Path | None = None, sources: list[str] | None = None) -> None:
    cfg = load_event(event_path)
    ensure_dirs(cfg)
    root = project_root()
    wanted = sources or list(INPUTS)
    for source in wanted:
        for src_name, dst_name in INPUTS.get(source, []):
            src = root / cfg["paths"]["data_dir"] / source / src_name
            dst = root / cfg["paths"]["data_dir"] / source / dst_name
            refine_one(src, dst, cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=None)
    parser.add_argument("--source", action="append", default=None)
    args = parser.parse_args()
    run(args.event, args.source)


if __name__ == "__main__":
    main()
