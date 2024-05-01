#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of xarray extensions. Currently, the following collections are attached to  xarray `DataArray` objects:
-  `gv` collects various methods that can be applied to individual DataArrays. A number of them are convenience plotting methods

The following are attached to xarray `Dataset` objects:
-  `gv` with miscellaneous Dataset methods. Note the same namespace as for the DataArray accessor above. This seems to work out okay.
-  `gadcp` collects ADCP-related methods. Mostly helpful for ADCP data processed with [velosearaptor](https://modscripps.github.io/velosearaptor/velosearaptor.html).

Accessors are a really neat way of attaching methods to xarray objects. Read more in the xarray docmumentation about [extending
xarray](https://docs.xarray.dev/en/stable/internals/extending-xarray.html).
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from pathlib import Path

import gvpy as gv


# Extend and/or modify xarray's DataArray capabilities
@xr.register_dataarray_accessor("gv")
class GunnarsAccessor:
    def __init__(self, xarray_obj):
        """This class collects a bunch of methods under `.gv`"""
        self._obj = xarray_obj
        self._center = None
        self._sampling_period = None
        if "lat" in self._obj.attrs:
            self.lat = self._obj.attrs["lat"]

    # @property
    # def center(self):
    #     """Return the geographic center point of this dataset.
    #     Example from the xarray docs.
    #     """
    #     if self._center is None:
    #         # we can use a cache on our accessor objects, because accessors
    #         # themselves are cached on instances that access them.
    #         lon = self._obj.latitude
    #         lat = self._obj.longitude
    #         self._center = (float(lon.mean()), float(lat.mean()))
    #     return self._center

    @property
    def sampling_period(self) -> float:
        """Sampling period in seconds (with three digit precision) if one of
        the dataset dimensions is time.

        Returns
        -------
        float
        """
        if self._sampling_period is None:
            if "time" in self._obj.dims:
                sampling_period_td = (
                    self._obj.time.diff("time").median().data.astype("timedelta64[ns]")
                )
                sampling_period_s = np.int64(
                    sampling_period_td.astype(np.float64) / 1e9
                )
                self._sampling_period = sampling_period_s.item()
        return self._sampling_period

    def tplot(self, **kwargs):
        """Quick time series plot.

        Returns
        -------
        ax
            Axis
        """

        def assign_cmap(t):
            # set a default cmap
            cmap = "viridis"
            cmap_dict = dict(
                # Spectral_r=[
                RdYlBu_r=[
                    "temperature",
                    "potential temperature",
                    "conservative temperature",
                    "in-situ temperature",
                    "temp",
                    "t",
                    "th",
                    "theta",
                    "Theta",
                    "$\\Theta$",
                ],
                RdBu_r=["u", "v", "w"],
            )

            longname = (
                t.attrs["long_name"] if "long_name" in t.attrs else "no_long_name"
            )
            for cmapi, longnames in cmap_dict.items():
                if longname in longnames:
                    cmap = cmapi
            return cmap

        def change_cf_labels():
            if "units" in self._obj.attrs:
                if self._obj.attrs["units"] == "m s-1":
                    self._obj.attrs["units"] = r"m$\,$s$^{-1}$"

        change_cf_labels()

        grid = kwargs.pop("grid", True)

        if "ax" not in kwargs:
            fig, ax = gv.plot.quickfig(w=8, h=3.5, grid=grid)
        else:
            ax = kwargs["ax"]
        if self._obj.ndim == 2 and "hue" not in kwargs:
            cbar_kwargs_new = dict(shrink=0.8, aspect=20, pad=0.01)
            if "cbar_kwargs" in kwargs:
                for k, v in kwargs["cbar_kwargs"].items():
                    cbar_kwargs_new[k] = v
            kwargs["cbar_kwargs"] = cbar_kwargs_new
            # this is hacky but allows to pass the add_colorbar argument
            if "add_colorbar" in kwargs and kwargs["add_colorbar"] is False:
                kwargs.pop("cbar_kwargs", True)

            kwargs["cmap"] = (
                assign_cmap(self._obj) if "cmap" not in kwargs else kwargs["cmap"]
            )
        if "hue" in kwargs and "add_legend" not in kwargs:
            kwargs["add_legend"] = False
        self._obj.plot(x="time", **kwargs)
        gv.plot.concise_date(ax, minticks=4)
        ax.set(xlabel="", title="")

        # determine whether the y-axis should be increasing
        invert_yaxis = False
        if "depth" in self._obj.dims:
            invert_yaxis = True
        if "pressure" in self._obj.dims:
            invert_yaxis = True
        if "p" in self._obj.dims:
            invert_yaxis = True
        if "z" in self._obj.dims and self._obj.z.median() > 0:
            invert_yaxis = True
        if "y" in kwargs:
            if kwargs["y"] == "hab":
                invert_yaxis = False
        if invert_yaxis:
            ax.invert_yaxis()

        xlab = ax.get_xlabel()
        if xlab[:4] == "time":
            ax.set(xlabel="")
        return ax

    def zplot(self, **kwargs):
        decrease_y = True
        grid = kwargs.pop("grid", True)
        if "ax" not in kwargs:
            fig, ax = gv.plot.quickfig(w=3.5, h=4, grid=grid)
        else:
            ax = kwargs["ax"]
        if "depth" in self._obj.coords:
            zvar = "depth"
        elif "z" in self._obj.coords:
            zvar = "z"
        ylabel = "depth [m]"
        if "y" in kwargs:
            zvar = kwargs.pop("y", True)
        if zvar == "hab":
            ylabel = "hab [m]"
            decrease_y = False

        self._obj.plot(y=zvar, **kwargs)
        ax.set(ylabel=ylabel, title="")
        if decrease_y:
            gv.plot.ydecrease(ax)
        return ax

    def llplot(self, **kwargs):
        """Lon-lat-plot with cartopy GeoAxes. Needs lon/lat coordinates.

        Returns
        -------
        ax : GeoAxes

        """
        da = self._obj
        grid = kwargs.pop("grid", True)
        if "fgs" in kwargs:
            fgs = kwargs.pop("fgs")
        else:
            fgs = (7, 7)
        if "ax" not in kwargs:
            projection = ccrs.Mercator()
            fig, ax = plt.subplots(
                figsize=fgs,
                subplot_kw={"projection": projection},
                constrained_layout=True,
                dpi=75,
            )
        else:
            ax = kwargs["ax"]

        vmin = da.min().data
        vmax = da.max().data
        if "vmin" in kwargs:
            vmin = kwargs.pop("vmin", vmin)
        if "vmax" in kwargs:
            vmax = kwargs.pop("vmax", vmax)

        h = da.plot(
            x="lon",
            y="lat",
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        # Adding the colorbar in a hacky way as the usual route for adding a
        # colorbar appears to conflict with cartopy, leading to over- or
        # undersized colorbars in many cases.

        # scale colorbar width by axis width
        pos = ax.get_position()
        cbar_width = 2 * 1 / pos.width

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right", size=f"{cbar_width}%", pad=0.08, axes_class=plt.Axes
        )
        cax.set_label("<colorbar>")

        if "long_name" in da.attrs:
            cbar_label = da.attrs["long_name"]
        else:
            cbar_label = da.name
        if "units" in da.attrs:
            cbar_label = cbar_label + f" [{da.attrs['units']}]"

        plt.colorbar(h, cax=cax, label=f"{cbar_label}")

        gv.plot.cartopy_axes(ax, maxticks=5)

        # No need for axis labels on a map
        ax.set(xlabel="", ylabel="")

        return ax

    def plot(self, **kwargs):
        """Shortcut for `tplot`"""
        if "time" in self._obj.coords and self._obj.time.size > 1:
            return self._obj.gv.tplot(**kwargs)
        elif "depth" in self._obj.coords or "z" in self._obj.coords:
            return self._obj.gv.zplot(**kwargs)
        elif "lon" in self._obj.coords and "lat" in self._obj.coords:
            return self._obj.gv.llplot(**kwargs)

    def tcoarsen(self, n=100):
        """Quickly coarsen DataArray along time dimension.

        Parameters
        ----------
        n : int
            Bin size for averaging.
        """
        return self._obj.coarsen(time=n, boundary="trim").mean()

    def plot_spectrum(self, N=None, nwind=2, lat=None):
        """Plot power spectral density with respect to cpd.

        Parameters
        ----------
        N : float, optional
            Buoyancy frequency used for calculating GM spectrum.

        nwind : int, optional
            Number of windows (more windows more smoothing). Defaults to 2.

        Returns
        -------
        ax

        """
        if lat is None:
            lat = self.lat

        f_cpd = gv.ocean.inertial_frequency(lat) / (2 * np.pi) * 3600 * 24

        # determine sampling period
        sp = self.sampling_period

        # calculate spectral density
        g = self._obj.data
        Pcw, Pccw, Ptot, omega = gv.signal.psd(
            g, sp, ffttype="t", window="hann", tser_window=g.size / nwind
        )

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(7, 5), constrained_layout=True
        )
        freqs = np.array(
            [
                24 / (14 * 24),
                24 / 12.4,
                2 * 24 / 12.4,
                4 * 24 / 12.4,
                f_cpd,
                2 * f_cpd,
                1,
            ]
        )
        freq_labels = ["fortnightly", "M2", "2M2", "4M2", " \nf", " \n2f", "K1"]
        for freq in freqs:
            ax.vlines(
                freq, 1e-3, 1e4, color="C0", alpha=0.5, linestyle="-", linewidth=0.75
            )

        # Spectrum
        ax.plot(omega * (3600 * 24) / (2 * np.pi), Ptot, linewidth=1, color="0.2")

        # GM
        if N is None:
            print(
                "No N provided, using N=2e-3 reflective of buoyancy frequency at ~1km depth"
            )
            N = 2e-3
        E = gv.gm81.calc_E_omg(N=N, lat=lat)
        ax.plot(E.omega * 3600 * 24 / (2 * np.pi), E.KE, label="KE", color="C3")

        # show -2 slope
        ax.plot([5e-2, 5e-1], [5e2, 5e0], color="C6")

        ax.set(xscale="log", yscale="log", xlim=(2.1e-2, 2e2), ylim=(1e-3, 1e5))
        ax = gv.plot.axstyle(ax, ticks="in", grid=True, spine_offset=10)
        gv.plot.gridstyle(ax, which="both")
        gv.plot.tickstyle(ax, which="both", direction="in")
        # ax2 = ax.twinx()
        ax2 = ax.secondary_xaxis(location="bottom")
        ax2 = gv.plot.axstyle(ax2, ticks="in", grid=False, spine_offset=30)
        ax2.xaxis.set_ticks([])
        ax2.xaxis.set_ticklabels([])
        ax2.minorticks_off()
        ax2.xaxis.set_ticks(freqs)
        ax2.xaxis.set_ticklabels(freq_labels)
        ax.set(ylabel="power spectral density [m$^2$/s$^2$/cps]")
        ax.set_xlabel("frequency [cpd]", labelpad=35)

        return ax

    def ts_hp(self, cutoff_period, order=3):
        """Time series high pass filter.
        Provide cutoff period in s.
        """
        cutoff_freq = 1 / cutoff_period
        fs = 1 / self.sampling_period
        axis = self._obj.get_axis_num("time")
        tmp = gv.signal.highpassfilter(
            self._obj, cutoff_freq, fs, order=order, axis=axis
        )
        out = self._obj.copy(data=tmp)
        if "long_name" in out.attrs:
            out.attrs["long_name"] = out.attrs["long_name"] + f" hp({cutoff_period}s)"
        return out

    def ts_lp(self, cutoff_period, order=3, type="butter"):
        """Time series low pass filter.
        Provide cutoff period in s.
        """
        cutoff_freq = 1 / cutoff_period
        fs = 1 / self.sampling_period
        axis = self._obj.get_axis_num("time")
        tmp = gv.signal.lowpassfilter(
            self._obj, cutoff_freq, fs, order=order, axis=axis, type=type
        )
        out = self._obj.copy(data=tmp)
        if "long_name" in out.attrs:
            out.attrs["long_name"] = out.attrs["long_name"] + f" lp({cutoff_period}s)"
        return out

    def to_netcdf(self, path, overwrite=True, confirm_overwrite=True):
        return _to_netcdf(self._obj, path, overwrite, confirm_overwrite)

    def duration(self, time_format="h"):
        return _duration(self._obj, time_format=time_format)


# Extend and/or modify xarray's Dataset capabilities.
# Naming this `gv` just as the DataArray accessor above.
# Let's hope this doesn't cause any issues.
@xr.register_dataset_accessor("gv")
class GunnarsDatasetAccessor:
    def __init__(self, xarray_obj):
        """This class collects a bunch of methods under `.gv`"""
        self._obj = xarray_obj
        # self.to_netcdf.__doc__ = _to_netcdf.__doc__

    def to_netcdf(self, path, overwrite=True, confirm_overwrite=True):
        return _to_netcdf(self._obj, path, overwrite, confirm_overwrite)

    def duration(self, time_format="h"):
        return _duration(self._obj, time_format=time_format)


# Add ADCP methods to xarray Dataset.
@xr.register_dataset_accessor("gadcp")
class GunnarsADCPAccessor:
    def __init__(self, xarray_obj):
        """This class collects a bunch of ADCP related methods under `.gadcp`.

        Assumes that ADCP data are in a Dataset structured by `velosearaptor`
        processing.
        """
        self._obj = xarray_obj

    def pg_filter(self, pg):
        """Remove ensemble averages smaller than a percent good threshold.
        Returns an adjusted copy of the dataset instead of filtering in place.

        Parameters
        ----------
        pg : float
            Percent good threshold.

        Returns
        -------
        ds : xr.Dataset
            ADCP dataset with u, v, w filtered.
        """
        ds = self._obj.copy()
        vars = ["u", "v", "w"]
        for var in vars:
            ds[var] = ds[var].where(ds.pg > pg)
        return ds


# Helper functions for cases where I want them accessible both in Datasets and DataArrays.
# These are wrapped with the accessor methods above.
def _to_netcdf(ds, path, overwrite=True, confirm_overwrite=True):
    """Wrapper for xarray's to_netcdf().

    Parameters
    ----------
    path : str or pathlib.Path()
        path can be just a filename (will be saved to current dir) or a
        Path() object defining a full path.
    overwrite : bool, optional
        Whether to overwrite an existing file at this location. Defaults to
        True.
    confirm_overwrite : bool, optional
        Whether to additionally confirm overwriting an existing file.
        Defaults to True.

    Returns
    -------
    path : pathlib.Path
        Path to netcdf file.

    Notes
    -----
    - If overwriting existing file, file will be deleted first. This is to
        get around the annoyance of files being locked that I have always
        been stupid to grasp and am now getting around with this. I am sure
        there would be a better and cleaner solution.
    - If time is a coordinate, it will be saved in cf-compliant format (seconds since 1970).
    """
    # note: the docstring above omits the first parameter, `ds`, since I am
    # wrapping this function with dataset and dataarray accessors and want to
    # be able to reuse the docstring.

    if isinstance(path, str):
        path = Path.cwd().joinpath(path)
    if not isinstance(path, Path):
        raise TypeError("Input must be a string or Path() object")
    if path.suffix == "":
        path = path.with_suffix(".nc")
    if path.suffix != ".nc":
        raise ValueError("File name must have either no suffix or end with '.nc'")

    exists = path.exists()
    if exists and not overwrite:
        print("File exists; set overwrite=True to replace")
        return

    if exists and overwrite:
        if confirm_overwrite:
            if gv.misc.yes_or_no("Confirm replacing existing file"):
                path.unlink()
            else:
                print("Aborting.")
                return
        else:
            path.unlink()

    opts = dict()

    if "time" in ds.coords:
        opts["encoding"] = {
            "time": {"units": "seconds since 1970-01-01", "dtype": "float"}
        }

    ds.to_netcdf(path, **opts)

    return path

def _duration(ds, time_format="h"):
    assert "time" in ds.coords, "no time coordinate"
    time = ds.time
    assert time.dtype.str[:3] == "<M8"
    dt = time[-1] - time[0]
    dt = dt.data.astype(f"<m8[{time_format}]")
    return dt


# Assign the original function's docstring to the wrapped method
GunnarsDatasetAccessor.to_netcdf.__doc__ = _to_netcdf.__doc__
GunnarsAccessor.to_netcdf.__doc__ = _to_netcdf.__doc__
