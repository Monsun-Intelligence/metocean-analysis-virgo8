#!/usr/bin/env python3
"""Monsun-style INACAWO meteograms (both sites + event marker).

Reads figures/point/inacawo/Metocean_Analysis_*.csv and writes
figures/point/inacawo/meteogram_{Hsig,PrimSwell,WindSpeed,Current,Hmax,SST,SSS}.png
matching the GFS/HYCOM plot_point chrome (event axvline, dual-site curves, direction arrows).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import ensure_dirs, load_event, project_root  # noqa: E402

SITE_FILES = {
    "incident": {
        "glob": "Metocean_Analysis_Hilang_Kontak*.csv",
        "color": "#DC2626",
        "short": "Hilang Kontak (Masalembo)",
    },
    "wreck": {
        "glob": "Metocean_Analysis_Estimasi_Drift*.csv",
        "color": "#0B2340",
        "short": "Estimasi Drift (~1.6 NM barat)",
    },
}

# CSV column → output plot
PARAMS = [
    {
        "key": "Hsig",
        "mag_col": "Hsig(m)",
        "dir_col": "WaveDir(deg)",
        "title": "Significant Wave Height",
        "ylabel": "Hsig (m)",
        "out": "meteogram_Hsig.png",
    },
    {
        "key": "PrimSwell",
        "mag_col": "PrimSwell(m)",
        "dir_col": "PrimSwellDir(deg)",
        "title": "Primary Swell",
        "ylabel": "PrimSwell (m)",
        "out": "meteogram_PrimSwell.png",
    },
    {
        "key": "WindSpeed",
        "mag_col": "WindSpeed(knots)",
        "dir_col": "WindDir(deg)",
        "title": "Wind Speed",
        "ylabel": "Wind (knots)",
        "out": "meteogram_WindSpeed.png",
    },
    {
        "key": "Current",
        "mag_col": "SurfCurrentSpd(cm/s)",
        "dir_col": "SurfCurrentDir",
        # INACAWO current direction is oceanographic "toward"
        "dir_convention": "to",
        "title": "Surface Current",
        "ylabel": "Current (cm/s)",
        "out": "meteogram_Current.png",
    },
    {
        "key": "Hmax",
        "mag_col": "Hmax(m)",
        "dir_col": "WaveDir(deg)",
        "title": "Maximum Wave Height",
        "ylabel": "Hmax (m)",
        "out": "meteogram_Hmax.png",
    },
    {
        "key": "SST",
        "mag_col": "SeaSurfaceTemperature(°C)",
        "dir_col": None,
        "title": "Sea Surface Temperature",
        "ylabel": "SST (°C)",
        "out": "meteogram_SST.png",
    },
    {
        "key": "SSS",
        "mag_col": "SeaSurfaceSalinity(PSU)",
        "dir_col": None,
        "title": "Sea Surface Salinity",
        "ylabel": "SSS (PSU)",
        "out": "meteogram_SSS.png",
    },
]


def _find_site_csv(inacawo_dir: Path, pattern: str) -> Path | None:
    matches = sorted(inacawo_dir.glob(pattern))
    # Prefer exact long names; skip combined/titik aggregates
    matches = [p for p in matches if "combined" not in p.name.lower() and "Titik_" not in p.name]
    return matches[0] if matches else None


def _dir_to_uv(
    direction_deg: np.ndarray, *, convention: str = "from"
) -> tuple[np.ndarray, np.ndarray]:
    """Direction (deg) → unit vectors (N↑ E→).

    convention='from' : meteorological (wind/wave coming from)
    convention='to'   : oceanographic (current flowing toward)
    """
    rad = np.deg2rad(np.asarray(direction_deg, dtype=float))
    if convention == "to":
        return np.sin(rad), np.cos(rad)
    return -np.sin(rad), -np.cos(rad)


def _unit_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.hypot(u, v)
    mag = np.where(mag > 1e-12, mag, np.nan)
    return u / mag, v / mag


def _load_sites(inacawo_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, meta in SITE_FILES.items():
        path = _find_site_csv(inacawo_dir, meta["glob"])
        if path is None:
            print(f"[SKIP] no CSV for {key} ({meta['glob']})")
            continue
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["Time(UTC/GMT)"])
        out[key] = df.sort_values("time")
        print(f"[OK] loaded {key}: {path.name} ({len(df)} rows)")
    return out


def _plot_param(
    sites: dict[str, pd.DataFrame],
    *,
    spec: dict,
    cfg: dict,
    out_path: Path,
) -> None:
    event_time = pd.Timestamp(cfg["event"]["time_utc"])
    if event_time.tzinfo is not None:
        event_time = event_time.tz_convert("UTC").tz_localize(None)

    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    any_data = False

    for site_key, df in sites.items():
        if spec["mag_col"] not in df.columns:
            continue
        meta = SITE_FILES[site_key]
        color = meta["color"]
        label = meta["short"]
        mag = df[spec["mag_col"]].to_numpy(dtype=float)
        t = pd.to_datetime(df["time"])
        ax.plot(t, mag, color=color, linewidth=0.85, label=label, zorder=3, alpha=0.8)
        any_data = True

        dir_col = spec.get("dir_col")
        if dir_col and dir_col in df.columns:
            uu, vv = _dir_to_uv(
                df[dir_col].to_numpy(dtype=float),
                convention=str(spec.get("dir_convention", "from")),
            )
            uu, vv = _unit_uv(uu, vv)
            t_arr = mdates.date2num(t.to_numpy())
            y_span = float(np.nanmax(mag) - np.nanmin(mag)) if len(mag) else 1.0
            y_span = y_span if y_span > 1e-6 else max(float(np.nanmax(mag)), 1.0)
            y_arrow = mag + 0.05 * y_span
            # Thin: every 2nd arrow for hourly series readability
            step = 2
            ax.quiver(
                t_arr[::step],
                y_arrow[::step],
                uu[::step],
                vv[::step],
                color=color,
                angles="uv",
                scale_units="height",
                scale=12.0,
                width=0.0009,
                headwidth=9.0,
                headlength=6.5,
                headaxislength=5.5,
                minshaft=1.2,
                pivot="middle",
                alpha=1.0,
                zorder=5,
            )

    if not any_data:
        plt.close(fig)
        print(f"[SKIP] no data for {spec['key']}")
        return

    event_label = f"Event ~{event_time.strftime('%H:%MZ')}"
    ax.axvline(
        event_time,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label=event_label,
        zorder=2,
    )

    ax.set_ylabel(spec["ylabel"], fontsize=10)
    ax.set_xlabel("UTC", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%HZ"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92, ncols=1)

    inc = cfg["event"]["incident"]
    wreck = cfg["event"].get("wreck") or {}
    subtitle = (
        f"{inc.get('label', 'Incident')}: {inc['lat']:.4f}°, {inc['lon']:.4f}°"
    )
    if wreck.get("lon") is not None:
        subtitle += (
            f"  ·  {wreck.get('label', 'Wreck')}: {wreck['lat']:.4f}°, {wreck['lon']:.4f}°"
        )

    arrow_note = "  ·  arrows = direction (N↑ E→)" if spec.get("dir_col") else ""
    ax.set_title(
        f"{spec['title']} — {cfg['event']['name']}\n"
        f"{subtitle}\n"
        f"Source: INACAWO (~3 km / 1 h){arrow_note}",
        fontsize=10,
        fontfamily="monospace",
        loc="left",
        pad=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(cfg["plot"]["dpi"]), facecolor="white")
    plt.close(fig)
    print(f"[OK] {out_path}")


def run(event_path: Path | None = None) -> list[Path]:
    cfg = load_event(event_path)
    ensure_dirs(cfg)
    root = project_root()
    inacawo_dir = root / cfg["paths"]["point_dir"] / "inacawo"
    sites = _load_sites(inacawo_dir)
    if not sites:
        print("[SKIP] no INACAWO point CSV")
        return []

    written: list[Path] = []
    for spec in PARAMS:
        out = inacawo_dir / spec["out"]
        _plot_param(sites, spec=spec, cfg=cfg, out_path=out)
        if out.exists():
            written.append(out)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=None)
    args = parser.parse_args()
    run(args.event)


if __name__ == "__main__":
    main()
