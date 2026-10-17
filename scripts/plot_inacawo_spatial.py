#!/usr/bin/env python3
"""Render Monsun-style spatial maps from INACAWO NetCDF day packages.

Expects data/inacawo/raw/{1013,1316}/ with Current, Wind, Average Wave, Swell.
Writes figures/spatial/inacawo/{current,wind,swh,swell}/t_YYYYMMDDHH.png
"""

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


def _first_var(ds: xr.Dataset) -> xr.DataArray:
    return ds[list(ds.data_vars)[0]]


def _open_chunk(chunk_dir: Path, rel: str) -> xr.DataArray:
    path = chunk_dir / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return _first_var(xr.open_dataset(path))


def _concat_chunks(raw_root: Path, rel: str) -> xr.DataArray:
    parts = []
    for name in ("1013", "1316"):
        chunk = raw_root / name
        if not chunk.exists():
            continue
        parts.append(_open_chunk(chunk, rel))
    if not parts:
        raise FileNotFoundError(f"No INACAWO chunks for {rel}")
    da = xr.concat(parts, dim="time").sortby("time")
    _, index = np.unique(da["time"].values, return_index=True)
    return da.isel(time=np.sort(index))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"[OK] {path}", flush=True)


def _dir_to_uv(direction: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological 'from' direction (deg) → unit propagation vectors."""
    rad = np.deg2rad(np.asarray(direction.values, dtype=float))
    return -np.sin(rad), -np.cos(rad)


def run(
    event_path: Path | None = None,
    every: int = 1,
    only: list[str] | None = None,
) -> None:
    cfg = load_event(event_path)
    ensure_dirs(cfg)
    root = project_root()
    raw = root / "data" / "inacawo" / "raw"
    out_root = root / cfg["paths"]["spatial_dir"] / "inacawo"
    markers = markers_from_event(cfg)
    bbox = plot_bbox(cfg)
    dpi = int(cfg["plot"]["dpi"])

    # Scalar + optional direction / vector components
    products = [
        {
            "key": "swh",
            "out": "swh",
            "mag_rel": "Average Wave/download_Wave.nc",
            "dir_rel": "Average Wave/download_WaveDir.nc",
            "mode": "dir",
            "scale": 1.0,
            "title": "Significant Wave Height",
            "source": "INACAWO (~3 km)",
        },
        {
            "key": "swell",
            "out": "swell",
            "mag_rel": "Swell/download_Swell.nc",
            "dir_rel": "Swell/download_SwellDir.nc",
            "mode": "dir",
            "scale": 1.0,
            "title": "Primary Swell",
            "source": "INACAWO (~3 km)",
        },
        {
            "key": "wind10m",
            "out": "wind",
            "mag_rel": "Wind/download_WindSpeed.nc",
            "u_rel": "Wind/download_WindSpeed_U.nc",
            "v_rel": "Wind/download_WindSpeed_V.nc",
            "mode": "uv",
            # Speeds already look like knots (0–25); keep map levels in kt
            "scale": 1.0,
            # Variable arrow length (matches GFS Atmos wind maps)
            "direction_only": False,
            "title": "10 m Wind",
            "source": "INACAWO (~3 km)",
        },
        {
            "key": "seacurrent",
            "out": "current",
            "mag_rel": "Current/download_CurrSpeed.nc",
            "u_rel": "Current/download_CurrSpeed_U.nc",
            "v_rel": "Current/download_CurrSpeed_V.nc",
            "mode": "uv",
            # Native m/s → cm/s for map levels
            "scale": 100.0,
            # Uniform arrow length; magnitude via color (matches HYCOM)
            "direction_only": True,
            "title": "Surface Current",
            "source": "INACAWO (~3 km)",
        },
    ]

    if only:
        wanted = {x.strip().lower() for x in only}
        products = [p for p in products if p["out"] in wanted]
        if not products:
            raise SystemExit(f"No products matched --only {only}")

    for spec in products:
        print(f"[INFO] INACAWO product {spec['out']}…", flush=True)
        mag = _concat_chunks(raw, spec["mag_rel"]) * float(spec["scale"])
        direction = uu = vv = None
        direction_only = bool(spec.get("direction_only", spec["mode"] == "dir"))
        if spec["mode"] == "dir":
            direction = _concat_chunks(raw, spec["dir_rel"])
            direction, mag = xr.align(direction, mag, join="inner")
            direction_only = True
        else:
            uu = _concat_chunks(raw, spec["u_rel"]) * float(spec["scale"])
            vv = _concat_chunks(raw, spec["v_rel"]) * float(spec["scale"])
            mag, uu, vv = xr.align(mag, uu, vv, join="inner")

        times = pd.DatetimeIndex(pd.to_datetime(mag["time"].values))
        if times.tz is not None:
            times = times.tz_convert("UTC").tz_localize(None)
        times = times[:: max(1, int(every))]
        prod = product_cfg(cfg, spec["key"])
        out_dir = out_root / spec["out"]

        for t in times:
            sl = mag.sel(time=t, method="nearest")
            valid = pd.Timestamp(sl["time"].values).to_pydatetime()
            stamp = valid.strftime("%Y%m%d%H")
            ui = vi = None
            if direction is not None:
                ui, vi = _dir_to_uv(direction.sel(time=t, method="nearest"))
            elif uu is not None and vv is not None:
                ui = uu.sel(time=t, method="nearest").values
                vi = vv.sel(time=t, method="nearest").values
            fig, _ = plot_scalar_map(
                sl["lon"].values,
                sl["lat"].values,
                sl.values,
                bbox=bbox,
                product=prod,
                title=spec["title"],
                valid_time=valid,
                source=spec["source"],
                markers=markers,
                densified=False,
                u=ui,
                v=vi,
                direction_only=direction_only,
                dpi=dpi,
            )
            _save(fig, out_dir / f"t_{stamp}.{cfg['plot']['fileformat']}")

        mag.close()
        if direction is not None:
            direction.close()
        if uu is not None:
            uu.close()
        if vv is not None:
            vv.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=None)
    parser.add_argument("--every", type=int, default=1, help="Plot every Nth hour")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Limit to product out-dirs: current wind swh swell",
    )
    args = parser.parse_args()
    run(args.event, every=args.every, only=args.only)


if __name__ == "__main__":
    main()
