#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of xarray extensions that can be found under .gv attached to xarray data objects, mostly to speed up stuff that otherwise I tediously have to type out."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import string
import xarray as xr
import bottleneck

from IPython import get_ipython

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
        if 'ax' not in kwargs:
            fig, ax = gvplot.quickfig()
        else:
            ax = kwargs['ax']
        if self._obj.ndim == 2:
            kwargs['cbar_kwargs'] = dict(shrink=0.8, aspect=25)
        self._obj.plot(x='time', **kwargs)
        gvplot.concise_date(ax, minticks=4)
        ax.set(xlabel='', title='')
        if 'depth' in self._obj.dims:
            ax.invert_yaxis()
        return ax

    def tcoarsen(self, n=100):
        return self._obj.coarsen(time=n, boundary='trim').mean()
