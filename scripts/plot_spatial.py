#!/usr/bin/env python3
"""Render Monsun-style spatial maps for SWH, swell, 10 m wind, and surface current."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import ensure_dirs, load_event, plot_bbox, project_root  # noqa: E402
from lib.plot_style import markers_from_event, plot_scalar_map, product_cfg  # noqa: E402
from lib.refine import refine_dataarray  # noqa: E402


def _prefer_refined(native: Path, refined: Path) -> Path:
    if refined.exists():
        return refined
    return native


def _times_to_plot(ds: xr.Dataset, cfg: dict) -> list[pd.Timestamp]:
    times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    every = int(cfg.get("plot", {}).get("every", 1))
    selected = list(times[::every])
    for key in cfg.get("plot", {}).get("key_times_utc", []):
        kt = pd.Timestamp(key)
        if kt.tzinfo is not None:
            kt = kt.tz_convert("UTC").tz_localize(None)
        idx = int(np.argmin(np.abs(times - kt)))
        t = pd.Timestamp(times[idx])
        if t not in selected:
            selected.append(t)
    return sorted(set(selected))


def _refine_slice(da: xr.DataArray, cfg: dict) -> xr.DataArray:
    if da.attrs.get("densified_target_deg") is not None:
        return da
    r = cfg["refine"]
    return refine_dataarray(
        da,
        target_deg=float(r["target_deg"]),
        method=r.get("method", "linear"),
        apply_gaussian=bool(r.get("apply_gaussian", True)),
        gaussian_sigma=float(r.get("gaussian_sigma", 0.8)),
        bbox=plot_bbox(cfg),
    )


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"[OK] {path}")


def plot_gfs_wave(cfg: dict) -> None:
    root = project_root()
    native = root / "data/gfs/gfs_wave_subset.nc"
    refined = root / "data/gfs/gfs_wave_refined.nc"
    path = _prefer_refined(native, refined)
    if not path.exists():
        print("[SKIP] GFS wave missing")
        return
    ds = xr.open_dataset(path)
    markers = markers_from_event(cfg)
    bbox = plot_bbox(cfg)
    densified = "refined" in path.name
    out_root = root / cfg["paths"]["spatial_dir"] / "gfs"

    for t in _times_to_plot(ds, cfg):
        sl = ds.sel(time=t, method="nearest")
        valid = pd.Timestamp(sl["time"].values).to_pydatetime()
        stamp = valid.strftime("%Y%m%d%H")

        # SWH
        if "swh" in sl:
            mag = _refine_slice(sl["swh"], cfg) if not densified else sl["swh"]
            u = v = None
            dir_only = True
            if "dirpw" in sl:
                direction = _refine_slice(sl["dirpw"], cfg) if not densified else sl["dirpw"]
                rad = np.deg2rad(direction.values)
                u = -np.sin(rad)
                v = -np.cos(rad)
            prod = product_cfg(cfg, "swh")
            fig, _ = plot_scalar_map(
                mag["lon"].values,
                mag["lat"].values,
                mag.values,
                bbox=bbox,
                product=prod,
                title=prod["display"],
                valid_time=valid,
                source="NOAA GFS Wave (archive)",
                markers=markers,
                densified=True,
                u=u,
                v=v,
                direction_only=dir_only,
                dpi=int(cfg["plot"]["dpi"]),
            )
            _save(fig, out_root / "swh" / f"t_{stamp}.{cfg['plot']['fileformat']}")

        # Swell
        if "swell" in sl:
            mag = _refine_slice(sl["swell"], cfg) if not densified else sl["swell"]
            u = v = None
            if "swdir" in sl:
                direction = _refine_slice(sl["swdir"], cfg) if not densified else sl["swdir"]
                rad = np.deg2rad(direction.values)
                u = -np.sin(rad)
                v = -np.cos(rad)
            prod = product_cfg(cfg, "swell")
            fig, _ = plot_scalar_map(
                mag["lon"].values,
                mag["lat"].values,
                mag.values,
                bbox=bbox,
                product=prod,
                title=prod["display"],
                valid_time=valid,
                source="NOAA GFS Wave (archive)",
                markers=markers,
                densified=True,
                u=u,
                v=v,
                direction_only=True,
                dpi=int(cfg["plot"]["dpi"]),
            )
            _save(fig, out_root / "swell" / f"t_{stamp}.{cfg['plot']['fileformat']}")
    ds.close()


def plot_gfs_wind(cfg: dict) -> None:
    root = project_root()
    native = root / "data/gfs/gfs_atmos_wind_subset.nc"
    refined = root / "data/gfs/gfs_atmos_wind_refined.nc"
    path = _prefer_refined(native, refined)
    if not path.exists():
        print("[SKIP] GFS atmos wind missing")
        return
    ds = xr.open_dataset(path)
    markers = markers_from_event(cfg)
    bbox = plot_bbox(cfg)
    densified = "refined" in path.name
    scale = float(cfg["products"]["wind10m"].get("scale", 1.94384))
    out_root = root / cfg["paths"]["spatial_dir"] / "gfs" / "wind10m"
    prod = product_cfg(cfg, "wind10m")

    for t in _times_to_plot(ds, cfg):
        sl = ds.sel(time=t, method="nearest")
        valid = pd.Timestamp(sl["time"].values).to_pydatetime()
        stamp = valid.strftime("%Y%m%d%H")
        u = _refine_slice(sl["u10"], cfg) if not densified else sl["u10"]
        v = _refine_slice(sl["v10"], cfg) if not densified else sl["v10"]
        mag = np.hypot(u.values, v.values) * scale
        fig, _ = plot_scalar_map(
            u["lon"].values,
            u["lat"].values,
            mag,
            bbox=bbox,
            product=prod,
            title=prod["display"],
            valid_time=valid,
            source="NOAA GFS Atmos (archive)",
            markers=markers,
            densified=True,
            u=u.values * scale,
            v=v.values * scale,
            direction_only=False,
            dpi=int(cfg["plot"]["dpi"]),
        )
        _save(fig, out_root / f"t_{stamp}.{cfg['plot']['fileformat']}")
    ds.close()


def plot_hycom_current(cfg: dict) -> None:
    root = project_root()
    native = root / "data/hycom/hycom_current_subset.nc"
    refined = root / "data/hycom/hycom_current_refined.nc"
    path = _prefer_refined(native, refined)
    if not path.exists():
        print("[SKIP] HYCOM current missing")
        return
    ds = xr.open_dataset(path)
    markers = markers_from_event(cfg)
    bbox = plot_bbox(cfg)
    densified = "refined" in path.name
    scale = float(cfg["products"]["seacurrent"].get("scale", 100.0))
    out_root = root / cfg["paths"]["spatial_dir"] / "hycom" / "seacurrent"
    prod = product_cfg(cfg, "seacurrent")

    for t in _times_to_plot(ds, cfg):
        sl = ds.sel(time=t, method="nearest")
        valid = pd.Timestamp(sl["time"].values).to_pydatetime()
        stamp = valid.strftime("%Y%m%d%H")
        u = _refine_slice(sl["water_u"], cfg) if not densified else sl["water_u"]
        v = _refine_slice(sl["water_v"], cfg) if not densified else sl["water_v"]
        mag = np.hypot(u.values, v.values) * scale
        fig, _ = plot_scalar_map(
            u["lon"].values,
            u["lat"].values,
            mag,
            bbox=bbox,
            product=prod,
            title=prod["display"],
            valid_time=valid,
            source="HYCOM ESPC-D-V02 archive",
            markers=markers,
            densified=True,
            u=u.values * scale,
            v=v.values * scale,
            direction_only=True,  # uniform arrow length; magnitude via color
            dpi=int(cfg["plot"]["dpi"]),
        )
        _save(fig, out_root / f"t_{stamp}.{cfg['plot']['fileformat']}")
    ds.close()


def run(event_path: Path | None = None, every: int | None = None) -> None:
    cfg = load_event(event_path)
    ensure_dirs(cfg)
    if every is not None:
        cfg.setdefault("plot", {})["every"] = every
    plot_gfs_wave(cfg)
    plot_gfs_wind(cfg)
    plot_hycom_current(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=None)
    parser.add_argument("--every", type=int, default=None, help="Plot every Nth timestep")
    args = parser.parse_args()
    run(args.event, every=args.every)


if __name__ == "__main__":
    main()
