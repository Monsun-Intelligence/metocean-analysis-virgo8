#!/usr/bin/env python3
"""Extract point time series (magnitude + direction) at incident and wreck sites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import ensure_dirs, load_event, project_root  # noqa: E402
from lib.extract import sample_point  # noqa: E402


def _to_utc_index(times) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(times))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def _dir_to_uv(dir_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological 'from' direction (°) → unit vector of propagation ('to')."""
    rad = np.deg2rad(np.asarray(dir_deg, dtype=float))
    return -np.sin(rad), -np.cos(rad)


def _vector_frame(
    ds: xr.Dataset,
    *,
    u_name: str,
    v_name: str,
    param: str,
    lon: float,
    lat: float,
    site: str,
    site_key: str,
    source: str,
) -> pd.DataFrame:
    u = sample_point(ds[u_name], lon, lat)
    v = sample_point(ds[v_name], lon, lat)
    uu = np.asarray(u.values, dtype=float).ravel()
    vv = np.asarray(v.values, dtype=float).ravel()
    mag = np.hypot(uu, vv)
    direction = (np.degrees(np.arctan2(uu, vv)) + 360.0) % 360.0
    return pd.DataFrame(
        {
            "time": _to_utc_index(u["time"].values),
            "param": param,
            "magnitude": mag,
            "u": uu,
            "v": vv,
            "direction_deg": direction,
            "lon": lon,
            "lat": lat,
            "site": site,
            "site_key": site_key,
            "source": source,
        }
    )


def _wave_frame(
    ds: xr.Dataset,
    *,
    mag_name: str,
    dir_name: str,
    param: str,
    lon: float,
    lat: float,
    site: str,
    site_key: str,
    source: str,
) -> pd.DataFrame:
    mag_da = sample_point(ds[mag_name], lon, lat)
    dir_da = sample_point(ds[dir_name], lon, lat)
    mag = np.asarray(mag_da.values, dtype=float).ravel()
    direction = np.asarray(dir_da.values, dtype=float).ravel()
    uu, vv = _dir_to_uv(direction)
    return pd.DataFrame(
        {
            "time": _to_utc_index(mag_da["time"].values),
            "param": param,
            "magnitude": mag,
            "u": uu,
            "v": vv,
            "direction_deg": direction,
            "lon": lon,
            "lat": lat,
            "site": site,
            "site_key": site_key,
            "source": source,
        }
    )


# One row per meteogram parameter
EXTRACTORS = [
    {
        "param": "swh",
        "source": "gfs",
        "file": "gfs_wave_subset.nc",
        "kind": "wave",
        "mag": "swh",
        "dir": "dirpw",
    },
    {
        "param": "swell",
        "source": "gfs",
        "file": "gfs_wave_subset.nc",
        "kind": "wave",
        "mag": "swell",
        "dir": "swdir",
    },
    {
        "param": "wind_sfc",
        "source": "gfs",
        "file": "gfs_wave_subset.nc",
        "kind": "vector",
        "u": "ugrdsfc",
        "v": "vgrdsfc",
    },
    {
        "param": "wind10m",
        "source": "gfs",
        "file": "gfs_atmos_wind_subset.nc",
        "kind": "vector",
        "u": "u10",
        "v": "v10",
    },
    {
        "param": "seacurrent",
        "source": "hycom",
        "file": "hycom_current_subset.nc",
        "kind": "vector",
        "u": "water_u",
        "v": "water_v",
    },
]


def run(event_path: Path | None = None) -> list[Path]:
    cfg = load_event(event_path)
    ensure_dirs(cfg)
    root = project_root()
    point_dir = root / cfg["paths"]["point_dir"]
    written: list[Path] = []

    sites = [("incident", cfg["event"]["incident"])]
    if cfg["event"].get("wreck"):
        sites.append(("wreck", cfg["event"]["wreck"]))

    frames: list[pd.DataFrame] = []
    # Cache open datasets by path
    open_ds: dict[Path, xr.Dataset] = {}

    try:
        for spec in EXTRACTORS:
            path = root / cfg["paths"]["data_dir"] / spec["source"] / spec["file"]
            if not path.exists():
                print(f"[SKIP] {path}")
                continue
            if path not in open_ds:
                open_ds[path] = xr.open_dataset(path)
            ds = open_ds[path]

            for site_key, site in sites:
                lon = float(site["lon"])
                lat = float(site["lat"])
                label = site.get("label", site_key)
                if spec["kind"] == "wave":
                    if spec["mag"] not in ds or spec["dir"] not in ds:
                        print(f"[SKIP] {spec['param']}: missing mag/dir in {path.name}")
                        continue
                    df = _wave_frame(
                        ds,
                        mag_name=spec["mag"],
                        dir_name=spec["dir"],
                        param=spec["param"],
                        lon=lon,
                        lat=lat,
                        site=label,
                        site_key=site_key,
                        source=spec["source"],
                    )
                else:
                    if spec["u"] not in ds or spec["v"] not in ds:
                        print(f"[SKIP] {spec['param']}: missing u/v in {path.name}")
                        continue
                    df = _vector_frame(
                        ds,
                        u_name=spec["u"],
                        v_name=spec["v"],
                        param=spec["param"],
                        lon=lon,
                        lat=lat,
                        site=label,
                        site_key=site_key,
                        source=spec["source"],
                    )
                out = point_dir / f"{spec['source']}_{spec['param']}_{site_key}_meteogram.csv"
                df.to_csv(out, index=False)
                written.append(out)
                frames.append(df)
                print(f"[OK] {out} rows={len(df)}")
    finally:
        for ds in open_ds.values():
            ds.close()

    if frames:
        combo = pd.concat(frames, ignore_index=True)
        combo = combo.sort_values(["param", "site_key", "time"]).reset_index(drop=True)
        combo_path = point_dir / "primary_meteogram.csv"
        combo.to_csv(combo_path, index=False)
        written.append(combo_path)
        print(f"[OK] {combo_path} rows={len(combo)}")

        # Legacy long-format CSV (magnitude only, incident + wreck) for notebook compat
        long_rows = combo.rename(columns={"magnitude": "value", "param": "variable"})[
            ["time", "variable", "value", "lon", "lat", "site", "source"]
        ].copy()
        # Map param names back to historical variable ids
        long_rows["variable"] = long_rows["variable"].replace(
            {
                "wind_sfc": "wind_sfc_ms",
                "wind10m": "wind10m_ms",
                "seacurrent": "seacurrent_ms",
            }
        )
        legacy = point_dir / "primary_incident_timeseries.csv"
        long_rows.to_csv(legacy, index=False)
        written.append(legacy)
        print(f"[OK] {legacy} (legacy long format)")

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=None)
    args = parser.parse_args()
    run(args.event)


if __name__ == "__main__":
    main()
