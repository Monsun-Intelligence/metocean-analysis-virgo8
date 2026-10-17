#!/usr/bin/env python3
"""Meteogram plots: one parameter per figure, both sites, magnitude + direction arrows."""

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

# Visual identity per site
SITE_STYLE = {
    "incident": {"color": "#DC2626", "short": "Hilang Kontak (Masalembo)"},
    "wreck": {"color": "#0B2340", "short": "Estimasi Drift (~1.6 NM barat)"},
}

PARAMS = [
    {
        "key": "swh",
        "title": "Significant Wave Height",
        "ylabel": "SWH (m)",
        "source_note": "NOAA GFS Wave",
    },
    {
        "key": "swell",
        "title": "Primary Swell",
        "ylabel": "Swell (m)",
        "source_note": "NOAA GFS Wave",
    },
    {
        "key": "wind10m",
        "title": "10 m Wind",
        "ylabel": "Wind (m/s)",
        "source_note": "NOAA GFS Atmos",
    },
    {
        "key": "wind_sfc",
        "title": "Wave-model Surface Wind",
        "ylabel": "Wind (m/s)",
        "source_note": "NOAA GFS Wave",
    },
    {
        "key": "seacurrent",
        "title": "Surface Current",
        "ylabel": "Current (m/s)",
        "source_note": "HYCOM ESPC-D-V02",
        "scale": 1.0,  # keep m/s; set 100 for cm/s if desired
    },
]


def _load_meteogram(point_dir: Path) -> pd.DataFrame:
    combo = point_dir / "primary_meteogram.csv"
    if combo.exists():
        return pd.read_csv(combo, parse_dates=["time"])
    frames = []
    for p in sorted(point_dir.glob("*_meteogram.csv")):
        frames.append(pd.read_csv(p, parse_dates=["time"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _unit_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.hypot(u, v)
    mag = np.where(mag > 1e-12, mag, np.nan)
    return u / mag, v / mag


def _plot_param(
    df: pd.DataFrame,
    *,
    spec: dict,
    cfg: dict,
    out_path: Path,
) -> None:
    sub = df[df["param"] == spec["key"]].copy()
    if sub.empty:
        print(f"[SKIP] no data for {spec['key']}")
        return

    scale = float(spec.get("scale", 1.0))
    event_time = pd.Timestamp(cfg["event"]["time_utc"])
    if event_time.tzinfo is not None:
        event_time = event_time.tz_convert("UTC").tz_localize(None)

    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)

    for site_key, g in sub.groupby("site_key", sort=False):
        g = g.sort_values("time")
        style = SITE_STYLE.get(site_key, {"color": "#6B7280", "short": site_key})
        color = style["color"]
        label = style["short"]
        mag = g["magnitude"].to_numpy(dtype=float) * scale
        t = pd.to_datetime(g["time"])
        ax.plot(t, mag, color=color, linewidth=0.85, label=label, zorder=3, alpha=0.8)

        uu, vv = _unit_uv(g["u"].to_numpy(dtype=float), g["v"].to_numpy(dtype=float))
        t_arr = mdates.date2num(t.to_numpy())
        # Lift arrows slightly above the curve so shafts/heads stay readable
        y_span = float(np.nanmax(mag) - np.nanmin(mag)) if len(mag) else 1.0
        y_span = y_span if y_span > 1e-6 else max(float(np.nanmax(mag)), 1.0)
        y_arrow = mag + 0.05 * y_span
        # Thin shaft (width) + larger head* multipliers so the tip stays visible
        ax.quiver(
            t_arr,
            y_arrow,
            uu,
            vv,
            color=color,
            angles="uv",
            scale_units="height",
            scale=12.0,  # larger → shorter arrows
            width=0.0009,
            headwidth=9.0,
            headlength=6.5,
            headaxislength=5.5,
            minshaft=1.2,
            pivot="middle",
            alpha=1.0,
            zorder=5,
        )

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
    if spec["key"] in ("swh", "swell"):
        reported = cfg["event"].get("reported_swh_m") or []
        if len(reported) >= 2:
            swh_lo, swh_hi = float(reported[0]), float(reported[1])
            ax.axhspan(
                swh_lo,
                swh_hi,
                color="#F59E0B",
                alpha=0.12,
                label=f"Reported {swh_lo:g}–{swh_hi:g} m",
                zorder=1,
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

    ax.set_title(
        f"{spec['title']} — {cfg['event']['name']}\n"
        f"{subtitle}\n"
        f"Source: {spec['source_note']}  ·  arrows = direction (N↑ E→)",
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
    point_dir = root / cfg["paths"]["point_dir"]
    df = _load_meteogram(point_dir)
    if df.empty:
        print("[SKIP] no meteogram CSV — run extract_point.py first")
        return []

    written: list[Path] = []
    out_dir = point_dir / "meteogram"
    for spec in PARAMS:
        out = out_dir / f"{spec['key']}.{cfg['plot']['fileformat']}"
        before = out.exists()
        _plot_param(df, spec=spec, cfg=cfg, out_path=out)
        if out.exists() and (before or out.stat().st_size > 0):
            # only append if plot wrote something (skip leaves stale file)
            if spec["key"] in set(df["param"].unique()):
                written.append(out)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=None)
    args = parser.parse_args()
    run(args.event)


if __name__ == "__main__":
    main()
