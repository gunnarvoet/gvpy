#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cartography."""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import scipy.ndimage as ndimage
from matplotlib.colors import (
    LightSource,
    LinearSegmentedColormap,
    ListedColormap,
)

try:
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
except ImportError:
    _has_cartopy = False
else:
    _has_cartopy = True


class HillShade:
    def __init__(self, topo, lon=None, lat=None, smoothtopo=5, shading=0.2):
        """Generate parameters for hill shading for an elevation model.

        Parameters
        ----------
        topo : array-like
            Topography
        lon : array-like, optional
            Longitude.
        lat : array-like, optional
            Latitude

        Notes
        -----
        With inspiration from this notebook:
        https://github.com/agile-geoscience/notebooks/blob/master/Colourmaps.ipynb
        """

        self.topo = topo
        if lon is not None and lat is not None:
            self.lon = lon
            self.lat = lat
            self.topo_extent = (
                self.lon.min(),
                self.lon.max(),
                self.lat.max(),
                self.lat.min(),
            )
        self.kmap = self.make_colormap([(0, 0, 0)])
        self.kmap4 = self.add_alpha(self.kmap)
        self.smoothbtopo = self.smooth_topo(sigma=smoothtopo)
        self.smoothbumps = self.generate_hill_shade(self.smoothbtopo, root=shading)

    def make_colormap(self, seq):
        """
        Converts a sequence of RGB tuples containing floats in the interval (0,1).
        For some reason LinearSegmentedColormap cannot take an alpha channel,
        even though matplotlib colourmaps have one.
        """
        seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
        cdict = {"red": [], "green": [], "blue": []}
        for i, item in enumerate(seq):
            if isinstance(item, float):
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]
                cdict["red"].append([item, r1, r2])
                cdict["green"].append([item, g1, g2])
                cdict["blue"].append([item, b1, b2])
        return LinearSegmentedColormap("CustomMap", cdict)

    def add_alpha(self, cmap, alpha=None):
        """
        Adds an alpha channel (opacity) to a colourmap. Uses a ramp by default.
        Pass an array of 256 values to use that. 0 = transparent; 1 = opaque.
        """
        cmap4 = cmap(np.arange(cmap.N))
        if alpha is None:
            alpha = np.linspace(1, 0, cmap.N)
        cmap4[:, -1] = alpha
        return ListedColormap(cmap4)

    def smooth_topo(self, sigma=2):
        """
        Smoothes topography using a gaussian filter in 2D.
        """
        stopo = ndimage.gaussian_filter(self.topo, sigma=(sigma, sigma), order=0)
        return stopo

    def generate_hill_shade(self, topo, root=1, azdeg=275, altdeg=145):
        """
        Generate image with hill shading.
        """
        ls = LightSource(azdeg=azdeg, altdeg=altdeg)
        bumps = ls.hillshade(topo) ** root  # Taking a root backs it off a bit.
        return bumps

    def plot_topo(self, ax, cmap="Blues"):
        mindepth = np.min(self.topo)
        maxdepth = np.max(self.topo)
        h = ax.contourf(
            self.lon,
            self.lat,
            self.topo,
            np.arange(mindepth, maxdepth, 100),
            cmap=cmap,
            vmin=mindepth,
            vmax=maxdepth + 500,
            extend="both",
            zorder=9,
        )
        for c in h.collections:
            c.set_rasterized(True)
            c.set_edgecolor("face")

        ax.imshow(
            self.smoothbumps,
            extent=self.topo_extent,
            cmap=self.kmap4,
            alpha=0.5,
            zorder=10,
        )

        # contour depth
        h2 = ax.contour(
            self.lon,
            self.lat,
            self.topo,
            np.arange(mindepth, maxdepth, 500),
            colors="0.1",
            linewidths=0.25,
            zorder=11,
        )
        for c in h2.collections:
            c.set_rasterized(True)
