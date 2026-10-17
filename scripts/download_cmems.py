#!/usr/bin/env python3
"""Optional Marine Copernicus (CMEMS) surface current download.

Requires `copernicusmarine` CLI/SDK credentials
(`copernicusmarine login` or COPERNICUSMARINE_* env vars).
Skips cleanly when unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import download_bbox, ensure_dirs, load_event, project_root  # noqa: E402

# Global ocean physics analysis/forecast — surface uo/vo
DEFAULT_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"


def run(event_path: Path | None = None) -> Path | None:
    cfg = load_event(event_path)
    ensure_dirs(cfg)
    try:
        import copernicusmarine
        import numpy as np
        import xarray as xr
    except ImportError as exc:
        print(f"[SKIP] CMEMS deps missing: {exc}")
        return None

    root = project_root()
    out = root / cfg["paths"]["data_dir"] / "cmems" / "cmems_current_subset.nc"
    if out.exists() and out.stat().st_size > 1000:
        print(f"[OK] CMEMS already present: {out}")
        return out

    lon0, lon1, lat0, lat1 = download_bbox(cfg)
    start = cfg["time_window"]["start"]
    end = cfg["time_window"]["end"]

    print(f"[INFO] Requesting CMEMS {DEFAULT_DATASET}…")
    try:
        ds = copernicusmarine.open_dataset(
            dataset_id=DEFAULT_DATASET,
            variables=["uo", "vo"],
            minimum_longitude=lon0,
            maximum_longitude=lon1,
            minimum_latitude=lat0,
            maximum_latitude=lat1,
            start_datetime=start,
            end_datetime=end,
        )
    except Exception as exc:
        print(f"[SKIP] CMEMS download failed (credentials/product?): {exc}")
        return None

    # Surface level if depth present
    if "depth" in ds.dims:
        ds = ds.isel(depth=0)
    rename = {}
    if "uo" in ds:
        rename["uo"] = "water_u"
    if "vo" in ds:
        rename["vo"] = "water_v"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    ds = ds.rename(rename)
    ds["speed"] = np.hypot(ds["water_u"], ds["water_v"])
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    print(f"[OK] Wrote {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=None)
    args = parser.parse_args()
    run(args.event)


if __name__ == "__main__":
    main()
