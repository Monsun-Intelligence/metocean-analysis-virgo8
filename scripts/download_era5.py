#!/usr/bin/env python3
"""Optional ERA5 download (CDS) for wind + waves over the analysis domain.

Requires ~/.cdsapirc and acceptance of ERA5 licenses.
Skips cleanly when credentials are missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import download_bbox, ensure_dirs, load_event, project_root  # noqa: E402


VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "significant_height_of_combined_wind_waves_and_swell",
    "significant_height_of_total_swell",
    "mean_direction_of_total_swell",
]


def cdsapirc() -> Path:
    return Path.home() / ".cdsapirc"


def run(event_path: Path | None = None) -> Path | None:
    cfg = load_event(event_path)
    ensure_dirs(cfg)
    if not cdsapirc().is_file():
        print("[SKIP] ERA5: missing ~/.cdsapirc — optional downloader not run")
        return None

    try:
        import cdsapi
        import xarray as xr
    except ImportError as exc:
        print(f"[SKIP] ERA5 deps missing: {exc}")
        return None

    root = project_root()
    out = root / cfg["paths"]["data_dir"] / "era5" / "era5_subset.nc"
    if out.exists() and out.stat().st_size > 1000:
        print(f"[OK] ERA5 already present: {out}")
        return out

    lon0, lon1, lat0, lat1 = download_bbox(cfg)
    # CDS area: North, West, South, East
    area = [lat1, lon0, lat0, lon1]
    start = cfg["time_window"]["start"][:10]
    end = cfg["time_window"]["end"][:10]

    import pandas as pd

    days = pd.date_range(start, end, freq="D")
    years = sorted({d.strftime("%Y") for d in days})
    months = sorted({d.strftime("%m") for d in days})
    day_list = sorted({d.strftime("%d") for d in days})
    hours = [f"{h:02d}:00" for h in range(24)]

    request = {
        "product_type": "reanalysis",
        "variable": VARIABLES,
        "year": years,
        "month": months,
        "day": day_list,
        "time": hours,
        "area": area,
        "format": "netcdf",
    }
    print("[INFO] Submitting ERA5 CDS request (may queue)…")
    client = cdsapi.Client()
    tmp = out.with_suffix(".download.nc")
    client.retrieve("reanalysis-era5-single-levels", request, str(tmp))

    ds = xr.open_dataset(tmp)
    rename = {
        "u10": "u10",
        "v10": "v10",
        "swh": "swh",
        "shts": "swell",
        "mdts": "swdir",
    }
    # CDS names vary; keep what exists
    keep = {}
    for src, dst in rename.items():
        if src in ds:
            keep[dst] = ds[src]
    # Also try long names / alternate short names
    for alt, dst in (
        ("Significant_height_of_combined_wind_waves_and_swell", "swh"),
        ("Significant_height_of_total_swell", "swell"),
    ):
        if alt in ds and dst not in keep:
            keep[dst] = ds[alt]
    out_ds = xr.Dataset(keep)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(out)
    tmp.unlink(missing_ok=True)
    print(f"[OK] Wrote {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=None)
    args = parser.parse_args()
    run(args.event)


if __name__ == "__main__":
    main()
