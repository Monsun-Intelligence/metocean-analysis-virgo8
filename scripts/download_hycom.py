#!/usr/bin/env python3
"""Download HYCOM ESPC-D-V02 surface currents via NCSS (day chunks).

OpenDAP annual aggregates often hang; NetcdfSubset day requests are reliable
for the analysis bbox.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import download_bbox, ensure_dirs, load_event, project_root  # noqa: E402
from lib.io_util import ensure_time_dim, normalize_lon_lat, save_dataset  # noqa: E402

NCSS_U = "https://ncss.hycom.org/thredds/ncss/grid/ESPC-D-V02/u3z/{year}"
NCSS_V = "https://ncss.hycom.org/thredds/ncss/grid/ESPC-D-V02/v3z/{year}"


def _ncss_url(
    base: str,
    varname: str,
    bbox: tuple[float, float, float, float],
    t0: pd.Timestamp,
    t1: pd.Timestamp,
) -> str:
    lon0, lon1, lat0, lat1 = bbox
    params = {
        "var": varname,
        "north": f"{lat1:.4f}",
        "south": f"{lat0:.4f}",
        "east": f"{lon1:.4f}",
        "west": f"{lon0:.4f}",
        "disableProjSubset": "on",
        "horizStride": "1",
        "time_start": pd.Timestamp(t0).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_end": pd.Timestamp(t1).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timeStride": "1",
        "vertCoord": "0",
        "accept": "netcdf4",
    }
    return f"{base}?{urllib.parse.urlencode(params)}"


def _download_ncss(url: str, dest: Path, retries: int = 6, timeout: int = 240) -> Path:
    """Fetch NCSS NetCDF via curl (more reliable than urllib/requests here)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.unlink(missing_ok=True)
        try:
            print(f"[INFO] NCSS curl ({attempt}/{retries}) {dest.name}", flush=True)
            cmd = [
                "curl",
                "-fsSL",
                "--retry",
                "2",
                "--retry-delay",
                "5",
                "--connect-timeout",
                "30",
                "--max-time",
                str(timeout),
                "-A",
                "monsun-hycom/1.0",
                "-o",
                str(tmp),
                url,
            ]
            subprocess.run(cmd, check=True)
            if tmp.stat().st_size < 1000:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"Download too small: {url}")
            head = tmp.read_bytes()[:200].lower()
            if b"<html" in head:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"NCSS returned error page: {url}")
            tmp.replace(dest)
            return dest
        except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
            last_err = exc
            tmp.unlink(missing_ok=True)
            dest.unlink(missing_ok=True)
            print(f"[WARN] {exc}", flush=True)
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"NCSS failed after {retries} tries: {last_err}")


def _load_component(path: Path, varname: str) -> xr.DataArray:
    ds = xr.open_dataset(path)
    # NCSS may nest depth; pick surface / squeeze
    if varname not in ds and varname.replace("water_", "") in ds:
        varname = varname.replace("water_", "")
    if varname not in ds:
        # first data var fallback
        candidates = [v for v in ds.data_vars if "water" in v.lower() or v.lower() in ("u", "v")]
        if not candidates:
            raise ValueError(f"{varname} missing in {path}; found {list(ds.data_vars)}")
        varname = candidates[0]
    da = ds[varname]
    if "depth" in da.dims:
        da = da.isel(depth=0)
    # Drop singleton dims except lon/lat/time
    da = da.squeeze(drop=True)
    da = da.rename(varname) if da.name != varname else da
    # Standardize coord names
    rename = {}
    for a, b in (("longitude", "lon"), ("latitude", "lat"), ("Longitude", "lon"), ("Latitude", "lat")):
        if a in da.coords or a in da.dims:
            rename[a] = b
    if rename:
        da = da.rename(rename)
    out = da.load()
    ds.close()
    return out


def _day_chunks(start: pd.Timestamp, end: pd.Timestamp):
    start = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tzinfo else pd.Timestamp(start)
    end = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo else pd.Timestamp(end)
    day = start.floor("D")
    while day <= end:
        t0 = max(start, day)
        t1 = min(end, day + pd.Timedelta(hours=21))
        if t0 <= t1:
            yield t0, t1
        day = day + pd.Timedelta(days=1)


def run(event_path: Path | None = None, skip_existing: bool = True) -> Path:
    cfg = load_event(event_path)
    ensure_dirs(cfg)
    root = project_root()
    data_dir = root / cfg["paths"]["data_dir"] / "hycom"
    raw_dir = data_dir / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "hycom_current_subset.nc"
    if skip_existing and out.exists() and out.stat().st_size > 1000:
        print(f"[OK] HYCOM already present: {out}", flush=True)
        return out

    start = pd.Timestamp(cfg["time_window"]["start"])
    end = pd.Timestamp(cfg["time_window"]["end"])
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    bbox = download_bbox(cfg)
    step = int(cfg["datasets"]["hycom"].get("step_hours", 3))

    u_parts: list[xr.DataArray] = []
    v_parts: list[xr.DataArray] = []
    for t0, t1 in _day_chunks(start, end):
        year = t0.year
        stamp = t0.strftime("%Y%m%d")
        u_path = raw_dir / f"u_{stamp}.nc"
        v_path = raw_dir / f"v_{stamp}.nc"
        _download_ncss(_ncss_url(NCSS_U.format(year=year), "water_u", bbox, t0, t1), u_path)
        _download_ncss(_ncss_url(NCSS_V.format(year=year), "water_v", bbox, t0, t1), v_path)
        u_parts.append(_load_component(u_path, "water_u"))
        v_parts.append(_load_component(v_path, "water_v"))
        print(f"[INFO] day {stamp} u times={u_parts[-1].sizes.get('time')} v times={v_parts[-1].sizes.get('time')}", flush=True)

    water_u = xr.concat(u_parts, dim="time").sortby("time")
    water_v = xr.concat(v_parts, dim="time").sortby("time")
    water_u, water_v = xr.align(water_u, water_v, join="inner")
    water_u = water_u.rename("water_u")
    water_v = water_v.rename("water_v")

    if step > 1 and water_u.sizes.get("time", 0) > 0:
        keep = [
            i
            for i, t in enumerate(pd.to_datetime(water_u["time"].values))
            if pd.Timestamp(t).hour % step == 0
        ]
        if keep:
            water_u = water_u.isel(time=keep)
            water_v = water_v.isel(time=keep)

    ds = xr.Dataset({"water_u": water_u, "water_v": water_v})
    ds["speed"] = np.hypot(ds["water_u"], ds["water_v"])
    ds = normalize_lon_lat(ds)
    ds = ensure_time_dim(ds)
    save_dataset(ds, out)
    print(f"[OK] Wrote {out} times={ds.sizes.get('time')}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.event, skip_existing=not args.force)


if __name__ == "__main__":
    main()
