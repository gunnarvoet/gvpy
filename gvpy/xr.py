#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of xarray extensions that can be found under .gv attached to xarray data objects, mostly to speed up stuff that otherwise I tediously have to type out."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import plot as gvplot


# extend and/or modify xarray's plotting capabilities
@xr.register_dataarray_accessor("gv")
class GunnarsAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._center = None

    @property
    def center(self):
        """Return the geographic center point of this dataset."""
        if self._center is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            lon = self._obj.latitude
            lat = self._obj.longitude
            self._center = (float(lon.mean()), float(lat.mean()))
        return self._center

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
                t.attrs["long_name"]
                if "long_name" in t.attrs
                else "no_long_name"
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
            fig, ax = gvplot.quickfig(w=8, h=3.5, grid=grid)
        else:
            ax = kwargs["ax"]
        if self._obj.ndim == 2 and "hue" not in kwargs:
            cbar_kwargs_new = dict(shrink=0.8, aspect=25)
            if "cbar_kwargs" in kwargs:
                for k, v in kwargs["cbar_kwargs"].items():
                    cbar_kwargs_new[k] = v
            kwargs["cbar_kwargs"] = cbar_kwargs_new

            kwargs["cmap"] = (
                assign_cmap(self._obj)
                if "cmap" not in kwargs
                else kwargs["cmap"]
            )
        if "hue" in kwargs and "add_legend" not in kwargs:
            kwargs["add_legend"] = False
        self._obj.plot(x="time", **kwargs)
        gvplot.concise_date(ax, minticks=4)
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

    # Just so can inject the .gv before plot() and don't have to type the tplot()...
    # Can still remove this if I want to make something different with it.
    def plot(self, **kwargs):
        self._obj.gv.tplot(**kwargs)

    def tcoarsen(self, n=100):
        return self._obj.coarsen(time=n, boundary="trim").mean()
