#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of xarray extensions.  Currently, methods are collected under
`.gv` and automatically attached to xarray `DataArray` objects.

Read more in the xarray docmumentation about [extending
xarray](https://docs.xarray.dev/en/stable/internals/extending-xarray.html).
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import gvpy as gv


# extend and/or modify xarray's plotting capabilities
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
    #     Just an example from the xarray docs.
    #     """
    #     if self._center is None:
    #         # we can use a cache on our accessor objects, because accessors
    #         # themselves are cached on instances that access them.
    #         lon = self._obj.latitude
    #         lat = self._obj.longitude
    #         self._center = (float(lon.mean()), float(lat.mean()))
    #     return self._center

    @property
    def sampling_period(self):
        """Return sampling period in seconds if one of the dimensions is time."""
        if self._sampling_period is None:
            if "time" in self._obj.dims:
                sampling_period_td = (
                    self._obj.time.diff("time").median().data.astype("timedelta64[s]")
                )
                sampling_period_s = sampling_period_td.astype(np.float64)
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
                Spectral_r=[
                    "temperature",
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
            cbar_kwargs_new = dict(shrink=0.8, aspect=25)
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
        if "depth" in self._obj.dims:
            ax.invert_yaxis()
        if "pressure" in self._obj.dims:
            ax.invert_yaxis()
        if "p" in self._obj.dims:
            ax.invert_yaxis()
        if "z" in self._obj.dims and self._obj.z.median() > 0:
            ax.invert_yaxis()
        xlab = ax.get_xlabel()
        if xlab[:4] == "time":
            ax.set(xlabel="")
        return ax

    def plot(self, **kwargs):
        """Shortcut for `tplot`"""
        self._obj.gv.tplot(**kwargs)

    def tcoarsen(self, n=100):
        """Quickly coarsen DataArray along time dimension.

        Parameters
        ----------
        n : int
            Bin size for averaging.
        """
        return self._obj.coarsen(time=n, boundary="trim").mean()

    def plot_spectrum(self, N=None, nwind=2):
        """Plot power spectral density

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

        f_cpd = gv.ocean.inertial_frequency(self.lat) / (2 * np.pi) * 3600 * 24

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
        E = gv.gm81.calc_E_omg(N=N, lat=self.lat)
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

