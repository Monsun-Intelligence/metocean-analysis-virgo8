"""Point extraction along native model timesteps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def _lon_lat_names(da: xr.DataArray) -> tuple[str, str]:
    lon_name = "lon" if "lon" in da.dims or "lon" in da.coords else "longitude"
    lat_name = "lat" if "lat" in da.dims or "lat" in da.coords else "latitude"
    return lon_name, lat_name


def _nearest_wet_indices(da2d: xr.DataArray, lon: float, lat: float) -> tuple[int, int]:
    lon_name, lat_name = _lon_lat_names(da2d)
    lonv = np.asarray(da2d[lon_name].values)
    latv = np.asarray(da2d[lat_name].values)
    vals = np.asarray(da2d.values)
    lon2d, lat2d = np.meshgrid(lonv, latv)
    dist = (lon2d - lon) ** 2 + (lat2d - lat) ** 2
    dist[~np.isfinite(vals)] = np.inf
    if not np.isfinite(dist).any():
        raise ValueError("No finite (wet) grid cells available for sampling")
    return np.unravel_index(int(np.argmin(dist)), dist.shape)


def sample_point(
    da: xr.DataArray,
    lon: float,
    lat: float,
    method: str = "linear",
) -> xr.DataArray:
    """Sample a field at (lon, lat), falling back to nearest wet cell if needed."""
    lon_name, lat_name = _lon_lat_names(da)
    sampled = da.interp({lon_name: lon, lat_name: lat}, method=method)

    # If all / any times are NaN (common on land-masked coastal points), use
    # nearest finite ocean cell based on the first time slice (or the 2-D field).
    values = np.asarray(sampled.values)
    if np.isfinite(values).any() and not np.isnan(values).all():
        # Still replace remaining NaNs time-by-time if mixed
        if np.isfinite(values).all():
            return sampled

    if "time" in da.dims:
        out = []
        times = da["time"]
        for t in range(da.sizes["time"]):
            sl = da.isel(time=t)
            try:
                j, i = _nearest_wet_indices(sl, lon, lat)
            except ValueError:
                out.append(np.nan)
                continue
            out.append(float(sl.values[j, i]))
        return xr.DataArray(
            out,
            coords={"time": times},
            dims=("time",),
            name=da.name,
            attrs=dict(da.attrs),
        )

    j, i = _nearest_wet_indices(da, lon, lat)
    return xr.DataArray(
        float(da.values[j, i]),
        name=da.name,
        attrs=dict(da.attrs),
    )


def series_to_dataframe(
    series: dict[str, xr.DataArray],
    *,
    lon: float,
    lat: float,
    label: str,
) -> pd.DataFrame:
    """Combine 1-D time DataArrays into a tidy DataFrame."""
    frames = []
    for name, da in series.items():
        if "time" in da.dims:
            times = pd.to_datetime(da["time"].values)
            values = np.asarray(da.values).ravel()
        else:
            times = pd.DatetimeIndex([pd.Timestamp.utcnow()])
            values = np.asarray([float(da.values)])
        frames.append(
            pd.DataFrame(
                {
                    "time": times,
                    "variable": name,
                    "value": values,
                    "lon": lon,
                    "lat": lat,
                    "site": label,
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["time", "variable", "value", "lon", "lat", "site"]
        )
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["variable", "time"]).reset_index(drop=True)
    return out


def extract_products_at_point(
    fields: dict[str, xr.DataArray],
    lon: float,
    lat: float,
    label: str = "incident",
    method: str = "linear",
) -> pd.DataFrame:
    sampled = {
        name: sample_point(da, lon, lat, method=method) for name, da in fields.items()
    }
    return series_to_dataframe(sampled, lon=lon, lat=lat, label=label)
