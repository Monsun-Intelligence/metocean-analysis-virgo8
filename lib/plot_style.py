"""Monsun-branded map chrome and discrete palettes (subset for this analysis)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

MS_ICE = "#E8F4FC"
MS_SKY = "#93C5FD"
MS_BLUE = "#0B74DE"
MS_NAVY = "#0B2340"
MS_CYAN = "#06B6D4"
MS_TEAL = "#14B8A6"
MS_MINT = "#6EE7B7"
MS_GOLD = "#F59E0B"
MS_CORAL = "#F97316"
MS_RED = "#DC2626"
MS_VIOLET = "#7C3AED"
MS_PLUM = "#581C87"

MONSUN_PALETTES = {
    "monsun_wind": [
        MS_ICE, "#C3E4FA", MS_SKY, "#60A5FA", MS_BLUE, MS_CYAN,
        MS_TEAL, MS_MINT, "#A3E635", MS_GOLD, MS_CORAL, MS_RED,
        MS_VIOLET, MS_PLUM,
    ],
    "monsun_wave": [
        "#DBEAFE", "#93C5FD", "#60A5FA", "#3B82F6", "#0EA5E9", MS_CYAN,
        MS_TEAL, "#34D399", "#A3E635", MS_GOLD, MS_CORAL, "#EA580C",
        MS_RED, "#DB2777",
    ],
    "monsun_swell": [
        "#E0F2FE", "#7DD3FC", "#38BDF8", "#2563EB", "#0EA5E9", MS_CYAN,
        MS_TEAL, "#5EEAD4", "#86EFAC", "#FDE047", MS_GOLD, MS_CORAL,
        MS_RED, "#C026D3",
    ],
    "monsun_current": [
        MS_ICE, MS_SKY, MS_CYAN, MS_TEAL, MS_MINT, "#4ADE80",
        MS_GOLD, MS_CORAL, MS_RED, MS_VIOLET, MS_PLUM, MS_NAVY,
    ],
}


def _resample_colors(colors: list[str], n: int) -> list[str]:
    if n <= 0 or len(colors) == n:
        return colors
    idx = np.linspace(0, len(colors) - 1, n)
    return [colors[int(round(i))] for i in idx]


def build_cmap_norm(palette: str, levels: list[float], extend: str = "max"):
    colors = list(MONSUN_PALETTES[palette])
    n_intervals = len(levels) - 1
    n_colors = n_intervals + (2 if extend == "both" else 1 if extend in ("min", "max") else 0)
    colors = _resample_colors(colors, n_colors)
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=len(colors), extend=extend)
    return cmap, norm


@dataclass
class Markers:
    incident_lon: float
    incident_lat: float
    incident_label: str
    wreck_lon: float | None = None
    wreck_lat: float | None = None
    wreck_label: str | None = None


def _panel(pad=1.8):
    return dict(
        boxstyle=f"round,pad={pad},rounding_size=0.4",
        facecolor="white",
        edgecolor="none",
        alpha=0.92,
    )


def plot_scalar_map(
    lon,
    lat,
    data,
    *,
    bbox: list[float],
    product: dict,
    title: str,
    valid_time: datetime,
    source: str,
    markers: Markers,
    densified: bool = True,
    u=None,
    v=None,
    direction_only: bool = False,
    figsize=(9.5, 9.5),
    dpi=120,
):
    """Render one Monsun-style edge-to-edge Mercator map."""
    levels = list(product["levels"])
    cmap, norm = build_cmap_norm(product["palette"], levels, product.get("extend", "max"))
    proj = ccrs.Mercator(
        central_longitude=0.5 * (bbox[0] + bbox[1]),
        min_latitude=bbox[2],
        max_latitude=bbox[3],
    )
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98], projection=proj)
    ax.set_extent(bbox, crs=ccrs.PlateCarree())
    try:
        ax.set_aspect("auto")
    except Exception:
        pass

    im = ax.contourf(
        lon,
        lat,
        data,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend=product.get("extend", "max"),
        transform=ccrs.PlateCarree(),
    )

    if u is not None and v is not None:
        # Thin vectors for readability on densified grids (~32 across domain).
        step = max(1, int(np.ceil(len(lon) / 32)))
        lon2d, lat2d = np.meshgrid(lon, lat)
        uu = np.asarray(u)[::step, ::step]
        vv = np.asarray(v)[::step, ::step]
        if direction_only:
            # Unit vectors + scale_units='width' → uniform visual arrow length
            # (magnitude is already encoded in the color field).
            mag = np.hypot(uu, vv)
            mag = np.where(mag > 1e-12, mag, np.nan)
            uu = uu / mag
            vv = vv / mag
            ax.quiver(
                lon2d[::step, ::step],
                lat2d[::step, ::step],
                uu,
                vv,
                transform=ccrs.PlateCarree(),
                angles="uv",
                scale_units="width",
                scale=36,  # larger → shorter arrows
                width=0.0022,
                headwidth=2.8,
                headlength=3.4,
                color="#0B2340",
                alpha=0.75,
                zorder=2.5,
            )
        else:
            ax.quiver(
                lon2d[::step, ::step],
                lat2d[::step, ::step],
                uu,
                vv,
                transform=ccrs.PlateCarree(),
                scale=400,
                width=0.003,
                color="#0B2340",
                alpha=0.75,
            )

    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#D1D5DB", zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor="#374151", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, edgecolor="#6B7280", linestyle="--", zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="white", alpha=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # Markers
    ax.plot(
        markers.incident_lon,
        markers.incident_lat,
        marker="*",
        markersize=14,
        color="#DC2626",
        markeredgecolor="white",
        markeredgewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=6,
        label=markers.incident_label,
    )
    if markers.wreck_lon is not None and markers.wreck_lat is not None:
        ax.plot(
            markers.wreck_lon,
            markers.wreck_lat,
            marker="o",
            markersize=7,
            color="#0B2340",
            markeredgecolor="white",
            markeredgewidth=0.8,
            transform=ccrs.PlateCarree(),
            zorder=6,
            label=markers.wreck_label or "Wreck",
        )
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    # Chrome panels
    densify_note = " · densified 0.01°" if densified else ""
    ax.text(
        0.012,
        0.985,
        f"{title}\n{product.get('region_label', '')}".strip(),
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        fontfamily="monospace",
        linespacing=1.2,
        zorder=10,
        bbox=_panel(),
    )
    ax.text(
        0.988,
        0.985,
        f"Valid: {valid_time.strftime('%b %d, %Y - %HUTC')}",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        fontfamily="monospace",
        zorder=10,
        bbox=_panel(),
    )
    ax.text(
        0.012,
        0.02,
        f"Source: {source}{densify_note}",
        transform=ax.transAxes,
        fontsize=7,
        ha="left",
        va="bottom",
        fontfamily="monospace",
        zorder=10,
    )
    ax.text(
        0.988,
        0.02,
        f"Monsun Analysis ©{datetime.now().year}",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        fontfamily="monospace",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#E5E7EB", edgecolor="none", alpha=0.9),
    )

    cbar_ax = inset_axes(
        ax,
        width="42%",
        height="2.2%",
        loc="lower left",
        bbox_to_anchor=(0.28, 0.035, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend=product.get("extend", "max"))
    cbar.ax.tick_params(labelsize=7)
    cbar_ax.set_facecolor("#E5E7EB")
    cbar_ax.text(
        1.02,
        0.5,
        product.get("unit", ""),
        transform=cbar_ax.transAxes,
        fontsize=8,
        va="center",
        ha="left",
        fontfamily="monospace",
    )

    # Bottom grey strip behind chrome
    strip = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        0.07,
        transform=ax.transAxes,
        boxstyle="square,pad=0",
        facecolor="#E5E7EB",
        edgecolor="none",
        alpha=0.55,
        zorder=5,
    )
    ax.add_patch(strip)
    return fig, ax


def product_cfg(event: dict, name: str) -> dict:
    cfg = dict(event["products"][name])
    cfg["region_label"] = event["domain"]["region_label"]
    return cfg


def markers_from_event(event: dict) -> Markers:
    inc = event["event"]["incident"]
    wreck = event["event"].get("wreck") or {}
    return Markers(
        incident_lon=float(inc["lon"]),
        incident_lat=float(inc["lat"]),
        incident_label=inc.get("label", "Incident"),
        wreck_lon=float(wreck["lon"]) if wreck.get("lon") is not None else None,
        wreck_lat=float(wreck["lat"]) if wreck.get("lat") is not None else None,
        wreck_label=wreck.get("label"),
    )


def ns(**kwargs):
    return SimpleNamespace(**kwargs)
