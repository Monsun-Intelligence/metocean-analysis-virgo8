"""Shared I/O helpers for subset NetCDF products."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def subset_lon_lat(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    lon0, lon1, lat0, lat1 = bbox
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon = ds[lon_name]
    # Handle 0–360 longitudes if needed.
    if float(lon.max()) > 180 and lon0 < 0:
        pass
    lat_slice = slice(lat0, lat1) if float(ds[lat_name][0]) < float(ds[lat_name][-1]) else slice(lat1, lat0)
    lon_slice = slice(lon0, lon1) if float(lon[0]) < float(lon[-1]) else slice(lon1, lon0)
    return ds.sel({lon_name: lon_slice, lat_name: lat_slice})


def normalize_lon_lat(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "longitude" in ds.coords or "longitude" in ds.dims:
        rename["longitude"] = "lon"
    if "latitude" in ds.coords or "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if rename:
        ds = ds.rename(rename)
    # Ensure ascending lat for plotting convenience.
    if "lat" in ds.coords and ds["lat"].size > 1:
        if float(ds["lat"][0]) > float(ds["lat"][-1]):
            ds = ds.sortby("lat")
    if "lon" in ds.coords and ds["lon"].size > 1:
        if float(ds["lon"][0]) > float(ds["lon"][-1]):
            ds = ds.sortby("lon")
    return ds


def ensure_time_dim(ds: xr.Dataset, time_val=None) -> xr.Dataset:
    if "time" in ds.dims:
        return ds
    if time_val is None:
        if "valid_time" in ds.coords:
            time_val = ds["valid_time"]
        elif "time" in ds.coords:
            time_val = ds["time"]
        else:
            time_val = np.datetime64("NaT")
    ds = ds.assign_coords(time=time_val)
    for name in list(ds.data_vars):
        if "time" not in ds[name].dims:
            ds[name] = ds[name].expand_dims("time")
    return ds


def clean_for_netcdf(ds: xr.Dataset) -> xr.Dataset:
    """Drop GRIB leftovers that break xarray NetCDF round-trips."""
    ds = ds.copy()
    drop = [
        name
        for name in (
            "step",
            "valid_time",
            "surface",
            "heightAboveGround",
            "orderedSequenceData",
        )
        if name in ds.variables or name in ds.coords
    ]
    if drop:
        ds = ds.drop_vars(drop, errors="ignore")
    for name in list(ds.variables):
        attrs = dict(ds[name].attrs)
        attrs.pop("dtype", None)
        for key in list(attrs):
            val = attrs[key]
            if isinstance(val, bool):
                attrs[key] = int(val)
            elif not isinstance(val, (str, int, float, bytes)) and val is not None:
                try:
                    float(val)
                except Exception:
                    attrs.pop(key, None)
        ds[name].attrs = attrs
    # Dataset-level attrs
    ds_attrs = {}
    for key, val in ds.attrs.items():
        if isinstance(val, bool):
            ds_attrs[key] = int(val)
        elif isinstance(val, (str, int, float, bytes)) or val is None:
            ds_attrs[key] = val
    ds.attrs = ds_attrs
    return ds


def save_dataset(ds: xr.Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_for_netcdf(ds)
    encoding = {name: {"zlib": True, "complevel": 4} for name in clean.data_vars}
    tmp = path.with_suffix(path.suffix + ".tmp")
    clean.to_netcdf(tmp, encoding=encoding)
    tmp.replace(path)
    return path


def open_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_timedelta=False)


def iter_times(ds: xr.Dataset):
    if "time" not in ds.coords:
        yield None, ds
        return
    times = pd.to_datetime(ds["time"].values)
    for i, t in enumerate(times):
        yield pd.Timestamp(t).to_pydatetime(), ds.isel(time=i)
