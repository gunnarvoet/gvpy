#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cartography."""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from matplotlib.colors import (
    LightSource,
    LinearSegmentedColormap,
    ListedColormap,
)

try:
    import cartopy
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
    import cartopy.geodesic as cgeo
    import shapely
except ImportError:
    _has_cartopy = False
else:
    _has_cartopy = True


class HillShade:
    def __init__(
        self, topo, lon=None, lat=None, smoothtopo=5, shading_factor=0.2
    ):
        """Generate parameters for hill shading for an elevation model.

        Parameters
        ----------
        topo : array-like
            Topography
        lon : array-like, optional
            Longitude.
        lat : array-like, optional
            Latitude
        smoothtopo : float, optional
            Smoothing factor for topography when calculating hillshading.
            Defaults to 5.
        shading_factor : float, optional
            Factor for hill shading. Less hill shading for a smaller factor.
            Defaults to 0.2.

        Attributes
        ----------
        smoothbtopo : array-like
            Smooth topography.
        smoothbumps : array-like
            Hill shades, based on `matplotlib.colors.LightSource`.
        kmap4 : matplotlib.colors.Colormap
            Black colormap with alpha channels for plotting hill shading.
        extent : `tuple`
            Extent in lon and lat. Only if coordinates are provided. Helpful
            for plotting.

        Notes
        -----
        With inspiration from this notebook:\n
        https://github.com/agile-geoscience/notebooks/blob/master/Colourmaps.ipynb

        The hill shading can be added on top of the topography like this:
        ```python
        hs = HillShade(topo, lon, lat)
        ax.imshow(
            hs.smoothbumps,
            extent=hs.topo_extent,
            cmap=hs.kmap4,
            alpha=0.5,
            zorder=10,
        )
        ```
        See code in `HillShade.plot_topo` for details.
        """

        self.topo = topo
        if 'lon' in topo:
            self.lon = topo.lon
            self.lat = topo.lat
        if lon is not None and lat is not None:
            self.lon = lon
            self.lat = lat
            self.topo_extent = (
                self.lon.min(),
                self.lon.max(),
                self.lat.max(),
                self.lat.min(),
            )
        self.kmap = self._make_colormap([(0, 0, 0)])
        self.kmap4 = self._add_alpha(self.kmap)
        self.smoothbtopo = self._smooth_topo(sigma=smoothtopo)
        self.smoothbumps = self._generate_hill_shade(
            self.smoothbtopo, root=shading_factor
        )

    def _make_colormap(self, seq):
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

    def _add_alpha(self, cmap, alpha=None):
        """
        Adds an alpha channel (opacity) to a colourmap. Uses a ramp by default.
        Pass an array of 256 values to use that. 0 = transparent; 1 = opaque.
        """
        cmap4 = cmap(np.arange(cmap.N))
        if alpha is None:
            alpha = np.linspace(1, 0, cmap.N)
        cmap4[:, -1] = alpha
        return ListedColormap(cmap4)

    def _smooth_topo(self, sigma=2):
        """
        Smoothes topography using a gaussian filter in 2D.
        """
        stopo = ndimage.gaussian_filter(
            self.topo, sigma=(sigma, sigma), order=0
        )
        return stopo

    def _generate_hill_shade(self, topo, root=1, azdeg=275, altdeg=145):
        """
        Generate image with hill shading.
        """
        ls = LightSource(azdeg=azdeg, altdeg=altdeg)
        bumps = ls.hillshade(topo) ** root  # Taking a root backs it off a bit.
        return bumps

    def plot_topo(self, ax, cmap="Blues"):
        """Plot topography with hill shading.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis instance for plotting.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for plotting. Defaults to "Blues".
        """
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

    def plot_topo_c(self, cmap="Blues"):
        """Plot topography with hill shading using cartopy.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis instance for plotting.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for plotting. Defaults to "Blues".
        """
        projection = ccrs.Mercator()

        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(8, 8),
            subplot_kw={"projection": projection},
        )

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
            zorder=2,
            transform=ccrs.PlateCarree(),
        )
        for c in h.collections:
            c.set_rasterized(True)
            c.set_edgecolor("face")

        ax.imshow(
            self.smoothbumps,
            extent=self.topo_extent,
            cmap=self.kmap4,
            alpha=0.5,
            zorder=3,
            transform=ccrs.PlateCarree(),
        )

        # contour depth
        h2 = ax.contour(
            self.lon,
            self.lat,
            self.topo,
            np.arange(mindepth, maxdepth, 500),
            colors="0.1",
            linewidths=0.25,
            zorder=4,
            transform=ccrs.PlateCarree(),
        )
        for c in h2.collections:
            c.set_rasterized(True)

        ax.set_extent(self.topo_extent, crs=ccrs.PlateCarree())

        return fig, ax


def cartopy_scale_bar(
    ax,
    location,
    length,
    metres_per_unit=1000,
    unit_name="km",
    tol=0.01,
    angle=0,
    color="black",
    linewidth=3,
    text_offset=0.005,
    ha="center",
    va="bottom",
    plot_kwargs=None,
    text_kwargs=None,
    **kwargs,
):
    """Scale bar for cartopy plots.
    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Args:
        ax:              CartoPy axes.
        location:        Position of left-side of bar in axes coordinates (x,y).
        length:          Geodesic length of the scale bar.
        metres_per_unit: Number of metres in the given unit. Default: 1000
        unit_name:       Name of the given unit. Default: 'km'
        tol:             Allowed relative error in length of bar. Default: 0.01
        angle:           Anti-clockwise rotation of the bar.
        color:           Color of the bar and text. Default: 'black'
        linewidth:       Same argument as for plot.
        text_offset:     Perpendicular offset for text in axes coordinates.
                         Default: 0.005
        ha:              Horizontal alignment. Default: 'center'
        va:              Vertical alignment. Default: 'bottom'
        **plot_kwargs:   Keyword arguments for plot, overridden by **kwargs.
        **text_kwargs:   Keyword arguments for text, overridden by **kwargs.
        **kwargs:        Keyword arguments for both plot and text.

    Notes
    -----
    [stackoverflow source code](https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot/50674451#50674451)
    """
    if not _has_cartopy:
        raise ImportError("Cartopy needs to be installed for this feature.")

    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {
        "linewidth": linewidth,
        "color": color,
        **plot_kwargs,
        **kwargs,
    }
    text_kwargs = {
        "ha": ha,
        "va": va,
        "rotation": angle,
        "color": color,
        **text_kwargs,
        **kwargs,
    }

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(
        ax, location, length_metres, angle=angle_rad, tol=tol
    )

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    h = ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ht = ax.text(
        *text_location,
        f"{length} {unit_name}",
        rotation_mode="anchor",
        transform=ax.transAxes,
        **text_kwargs,
    )
    return h, ht


def plot_watch_circle(lon, lat, radius, ax, zorder=50, ec="b", alpha=1):
    """Plot a watch circle on a cartopy map.

    Parameters
    ----------
    lon : float
        Longitude
    lat : float
        Latitude
    radius : float
        Circle radius in m.
    ax : matplotlib.axes.Axes
        Axis instance for plotting.
    zorder : int
        Vertical order on matplotlib plot. Optional, defaults to 50 (pretty high).
    ec : color or None or 'auto'
        Edgecolor. Optional, defaults to 'b'.
    """
    if not _has_cartopy:
        raise ImportError("Cartopy needs to be installed for this feature.")
    circle_points = cartopy.geodesic.Geodesic().circle(
        lon,
        lat,
        radius=radius,
        n_samples=100,
        endpoint=False,
    )
    geom = shapely.geometry.Polygon(circle_points)
    ax.add_geometries(
        (geom,),
        crs=cartopy.crs.PlateCarree(),
        facecolor="none",
        edgecolor=ec,
        linewidth=1,
        zorder=zorder,
        alpha=alpha,
    )


def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        direction  Nonzero (2, 1)-shaped array, a direction vector.
        distance:  Positive distance to go past.
        dist_func: A two-argument function which returns distance.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment  from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        end:       Outer bound on point's location.
        distance:  Positive distance to travel.
        dist_func: Two-argument function which returns distance.
        tol:       Relative error in distance to allow.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(
            f"End is closer to start ({initial_distance}) than "
            f"given distance ({distance})."
        )

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Args:
        ax:       CartoPy axes.
        start:    Starting point for the line in axes coordinates.
        distance: Positive physical distance to travel.
        angle:    Anti-clockwise angle for the bar, in radians. Default: 0
        tol:      Relative error in distance to allow. Default: 0.01

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cgeo.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        return geodesic.inverse(a_phys, b_phys)[0][0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)
