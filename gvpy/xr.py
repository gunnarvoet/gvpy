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
            cmap = 'viridis'
            cmap_dict = dict(
                    Spectral_r = ['temperature', 'temp', 't', 'th', 'theta', 'Theta', '$\\Theta$'],
                    RdBu_r =['u', 'v'],
                    )

            longname = t.attrs['long_name'] if 'long_name' in t.attrs else 'no_long_name'
            for cmapi, longnames in cmap_dict.items():
                if longname in longnames:
                    cmap = cmapi
            return cmap

        if 'ax' not in kwargs:
            fig, ax = gvplot.quickfig(w=8, h=3.5)
        else:
            ax = kwargs['ax']
        if self._obj.ndim == 2:
            kwargs['cbar_kwargs'] = dict(shrink=0.8, aspect=25)
            kwargs['cmap'] = assign_cmap(self._obj) if 'cmap' not in kwargs else kwargs['cmap']
        self._obj.plot(x='time', **kwargs)
        gvplot.concise_date(ax, minticks=4)
        ax.set(xlabel='', title='')
        if 'depth' in self._obj.dims:
            ax.invert_yaxis()
        if 'z' in self._obj.dims and self._obj.z.median()>0:
            ax.invert_yaxis()
        return ax

    def tcoarsen(self, n=100):
        return self._obj.coarsen(time=n, boundary='trim').mean()
