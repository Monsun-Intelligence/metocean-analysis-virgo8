"""Spatial densification for coarse model fields (display only)."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter


def _lon_lat_names(da: xr.DataArray) -> tuple[str, str]:
    lon_name = "lon" if "lon" in da.dims else "longitude"
    lat_name = "lat" if "lat" in da.dims else "latitude"
    if lon_name not in da.dims or lat_name not in da.dims:
        raise ValueError(f"Expected lon/lat dims on {da.name}, got {da.dims}")
    return lon_name, lat_name


def build_target_coords(
    da: xr.DataArray,
    target_deg: float,
    bbox: list[float] | None = None,
) -> dict[str, np.ndarray]:
    lon_name, lat_name = _lon_lat_names(da)
    lon = np.asarray(da[lon_name].values)
    lat = np.asarray(da[lat_name].values)
    if bbox is not None:
        lon0, lon1, lat0, lat1 = bbox
        lon_min, lon_max = lon0, lon1
        lat_min, lat_max = lat0, lat1
    else:
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    new_lon = np.arange(lon_min, lon_max + target_deg * 0.5, target_deg)
    new_lat = np.arange(lat_min, lat_max + target_deg * 0.5, target_deg)
    return {lon_name: new_lon, lat_name: new_lat}


def _gaussian_2d(values: np.ndarray, sigma: float) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim < 2:
        return out
    # Apply over trailing lat/lon axes; preserve leading dims (e.g. time).
    flat = out.reshape((-1,) + out.shape[-2:])
    masked = np.ma.masked_invalid(flat)
    fill = masked.filled(0.0)
    weight = (~masked.mask).astype(float)
    blur = gaussian_filter(fill, sigma=(0.0, sigma, sigma))
    wblur = gaussian_filter(weight, sigma=(0.0, sigma, sigma))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = blur / np.where(wblur > 1e-6, wblur, np.nan)
    result[wblur <= 1e-6] = np.nan
    return result.reshape(out.shape)


def refine_dataarray(
    da: xr.DataArray,
    *,
    target_deg: float = 0.01,
    method: str = "linear",
    apply_gaussian: bool = True,
    gaussian_sigma: float = 0.8,
    bbox: list[float] | None = None,
) -> xr.DataArray:
    """Interpolate to a finer lon/lat grid, optionally Gaussian-smooth."""
    coords = build_target_coords(da, target_deg, bbox=bbox)
    refined = da.interp(coords, method=method)
    if apply_gaussian and gaussian_sigma and gaussian_sigma > 0:
        smoothed = _gaussian_2d(refined.values, float(gaussian_sigma))
        refined = refined.copy(data=smoothed)
    refined.attrs.update(da.attrs)
    refined.attrs["densified_target_deg"] = target_deg
    refined.attrs["densified_note"] = (
        "Spatially densified for display; does not add sub-grid physics."
    )
    return refined


def refine_dataset(
    ds: xr.Dataset,
    variables: list[str] | None = None,
    **kwargs,
) -> xr.Dataset:
    names = variables or list(ds.data_vars)
    out = {}
    for name in names:
        if name not in ds:
            continue
        out[name] = refine_dataarray(ds[name], **kwargs)
    result = xr.Dataset(out)
    result.attrs.update({k: v for k, v in ds.attrs.items() if not isinstance(v, bool)})
    result.attrs["densified"] = 1
    return result
