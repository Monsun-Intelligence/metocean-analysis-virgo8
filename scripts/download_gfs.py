#!/usr/bin/env python3
"""Download GFS Wave + Atmos archive fields for the event window.

Uses AWS Open Data bucket noaa-gfs-bdp-pds (NOMADS retention is too short).
Wave: global 0.25° GRIB → subset locally.
Atmos 10 m wind: uses GRIB .idx byte-range subset when possible.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import download_bbox, ensure_dirs, load_event, project_root  # noqa: E402
from lib.io_util import ensure_time_dim, normalize_lon_lat, save_dataset, subset_lon_lat  # noqa: E402

AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


def _cycles_and_fxx(start: pd.Timestamp, end: pd.Timestamp, step_hours: int = 3):
    """Yield (cycle_str YYYYMMDDHH, fxx, valid_time) covering [start, end] at step_hours."""
    # Align to 3-hourly valids; use nearest prior 00/06/12/18 cycle with fxx in 0..5.
    valid = start.floor(f"{step_hours}h")
    if valid < start:
        valid = valid + pd.Timedelta(hours=step_hours)
    end = pd.Timestamp(end)
    while valid <= end:
        cycle_hour = (valid.hour // 6) * 6
        cycle = valid.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        # If valid hour is before cycle within same day issue — shouldn't happen.
        fxx = int((valid - cycle).total_seconds() // 3600)
        if fxx < 0:
            cycle = cycle - pd.Timedelta(hours=6)
            fxx = int((valid - cycle).total_seconds() // 3600)
        yield cycle.strftime("%Y%m%d%H"), fxx, valid.to_pydatetime().replace(tzinfo=None)
        valid = valid + pd.Timedelta(hours=step_hours)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    print(f"[INFO] Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size < 1000:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download too small: {url}")
    return dest


def wave_urls(cycle: str, fxx: int) -> str:
    y, m, d, h = cycle[:4], cycle[4:6], cycle[6:8], cycle[8:10]
    fname = f"gfswave.t{h}z.global.0p25.f{fxx:03d}.grib2"
    return f"{AWS_BASE}/gfs.{y}{m}{d}/{h}/wave/gridded/{fname}"


def atmos_urls(cycle: str, fxx: int) -> tuple[str, str]:
    y, m, d, h = cycle[:4], cycle[4:6], cycle[6:8], cycle[8:10]
    base = f"{AWS_BASE}/gfs.{y}{m}{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f{fxx:03d}"
    return base, f"{base}.idx"


def _normalize_coords(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "longitude" in da.coords:
        rename["longitude"] = "lon"
    if "latitude" in da.coords:
        rename["latitude"] = "lat"
    return da.rename(rename) if rename else da


def normalize_gfswave(ds: xr.Dataset) -> xr.Dataset:
    out = {}
    if "u" in ds:
        out["ugrdsfc"] = _normalize_coords(ds["u"])
    if "v" in ds:
        out["vgrdsfc"] = _normalize_coords(ds["v"])
    if "swh" in ds:
        out["swh"] = _normalize_coords(ds["swh"])
    if "dirpw" in ds:
        out["dirpw"] = _normalize_coords(ds["dirpw"])
    # Primary swell (first ordered sequence)
    if "shts" in ds:
        shts = ds["shts"]
        if "orderedSequenceData" in shts.dims:
            shts = shts.isel(orderedSequenceData=0)
        out["swell"] = _normalize_coords(shts)
    if "swdir" in ds:
        swdir = ds["swdir"]
        if "orderedSequenceData" in swdir.dims:
            swdir = swdir.isel(orderedSequenceData=0)
        out["swdir"] = _normalize_coords(swdir)
    if not out:
        raise ValueError(f"No GFS Wave vars found: {list(ds.data_vars)}")
    result = xr.Dataset(out)
    return ensure_time_dim(normalize_lon_lat(result), ds.get("valid_time", ds.get("time")))


def normalize_gfsatmos_wind(ds: xr.Dataset) -> xr.Dataset:
    out = {}
    if "u10" in ds and "v10" in ds:
        out["u10"] = _normalize_coords(ds["u10"])
        out["v10"] = _normalize_coords(ds["v10"])
    elif "u" in ds and "v" in ds:
        u, v = ds["u"], ds["v"]
        if "heightAboveGround" in u.coords:
            u = u.sel(heightAboveGround=10, method="nearest")
            v = v.sel(heightAboveGround=10, method="nearest")
        out["u10"] = _normalize_coords(u)
        out["v10"] = _normalize_coords(v)
    if not out:
        raise ValueError(f"No 10 m wind in atmos GRIB: {list(ds.data_vars)}")
    result = xr.Dataset(out)
    return ensure_time_dim(normalize_lon_lat(result), ds.get("valid_time", ds.get("time")))


def _parse_idx(idx_text: str) -> list[dict]:
    rows = []
    for line in idx_text.splitlines():
        parts = line.split(":")
        if len(parts) < 7:
            continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        rows.append(
            {
                "offset": offset,
                "param": parts[3],
                "level": parts[4],
                "raw": line,
            }
        )
    # Attach end offsets
    for i, row in enumerate(rows):
        row["end"] = rows[i + 1]["offset"] if i + 1 < len(rows) else None
    return rows


def download_atmos_uv10(cycle: str, fxx: int, cache_dir: Path) -> Path:
    """Byte-range download of UGRD/VGRD 10 m messages via .idx."""
    grib_url, idx_url = atmos_urls(cycle, fxx)
    out = cache_dir / f"gfsatmos_{cycle}_f{fxx:03d}_uv10.grib2"
    if out.exists() and out.stat().st_size > 500:
        return out

    print(f"[INFO] Fetching idx {idx_url}")
    with urllib.request.urlopen(idx_url, timeout=120) as resp:
        idx_text = resp.read().decode("utf-8", errors="replace")
    rows = _parse_idx(idx_text)
    wanted = [
        r
        for r in rows
        if r["param"] in ("UGRD", "VGRD") and "10 m above ground" in r["level"]
    ]
    if len(wanted) < 2:
        raise RuntimeError(f"Could not find UGRD/VGRD 10 m in idx for {cycle} f{fxx:03d}")

    chunks = []
    for row in wanted:
        req = urllib.request.Request(grib_url)
        end = row["end"] - 1 if row["end"] is not None else ""
        req.add_header("Range", f"bytes={row['offset']}-{end}")
        print(f"[INFO] Range-get {row['param']} {row['level']} @ {row['offset']}")
        with urllib.request.urlopen(req, timeout=180) as resp:
            chunks.append(resp.read())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"".join(chunks))
    return out


def open_grib(path: Path) -> xr.Dataset:
    parts = cfgrib.open_datasets(str(path))
    return xr.merge(parts, compat="override")


def run(event_path: Path | None = None, skip_existing: bool = True) -> Path:
    cfg = load_event(event_path)
    ensure_dirs(cfg)
    root = project_root()
    data_dir = root / cfg["paths"]["data_dir"] / "gfs"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(cfg["time_window"]["start"])
    end = pd.Timestamp(cfg["time_window"]["end"])
    bbox = download_bbox(cfg)
    step = int(cfg["datasets"]["gfs"]["wave_step_hours"])

    for cycle, fxx, valid in _cycles_and_fxx(start, end, step_hours=step):
        stamp = valid.strftime("%Y%m%d%H")
        # --- Wave ---
        wave_nc = data_dir / f"wave_{stamp}.nc"
        if not (skip_existing and wave_nc.exists()):
            try:
                grib_path = raw_dir / f"gfswave_{cycle}_f{fxx:03d}.grib2"
                _download(wave_urls(cycle, fxx), grib_path)
                raw = open_grib(grib_path)
                ds = normalize_gfswave(raw)
                ds = subset_lon_lat(ds, bbox)
                ds = ds.assign_coords(time=("time", [np.datetime64(valid)]))
                save_dataset(ds, wave_nc)
                raw.close()
                ds.close()
            except Exception as exc:
                print(f"[WARN] Wave {cycle} f{fxx:03d}: {exc}")

        # --- Atmos 10 m wind ---
        atmos_nc = data_dir / f"atmos_{stamp}.nc"
        if not (skip_existing and atmos_nc.exists()):
            try:
                grib_path = download_atmos_uv10(cycle, fxx, raw_dir)
                raw = open_grib(grib_path)
                ds = normalize_gfsatmos_wind(raw)
                ds = subset_lon_lat(ds, bbox)
                ds = ds.assign_coords(time=("time", [np.datetime64(valid)]))
                save_dataset(ds, atmos_nc)
                raw.close()
                ds.close()
            except Exception as exc:
                print(f"[WARN] Atmos {cycle} f{fxx:03d}: {exc}")

    wave_files = sorted(data_dir.glob("wave_*.nc"))
    if not wave_files:
        raise RuntimeError("No GFS Wave frames downloaded")

    wave_all = xr.concat(
        [xr.open_dataset(p, decode_timedelta=False) for p in wave_files],
        dim="time",
    ).sortby("time")
    wave_out = data_dir / "gfs_wave_subset.nc"
    save_dataset(wave_all, wave_out)
    wave_all.close()
    print(f"[OK] Wrote {wave_out}")

    atmos_files = sorted(data_dir.glob("atmos_*.nc"))
    if atmos_files:
        atmos_all = xr.concat(
            [xr.open_dataset(p, decode_timedelta=False) for p in atmos_files],
            dim="time",
        ).sortby("time")
        atmos_out = data_dir / "gfs_atmos_wind_subset.nc"
        save_dataset(atmos_all, atmos_out)
        atmos_all.close()
        print(f"[OK] Wrote {atmos_out}")

    return wave_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Re-download even if NetCDF exists")
    args = parser.parse_args()
    run(args.event, skip_existing=not args.force)


if __name__ == "__main__":
    main()
