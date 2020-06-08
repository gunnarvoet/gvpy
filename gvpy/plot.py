#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Anything plotting related (mostly matplotlib) lives here."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from IPython import get_ipython

ipython = get_ipython()

# import cm to register colormaps defined therein
from . import cm

try:
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
except ImportError:
    _has_cartopy = False
else:
    _has_cartopy = True


def nostalgic():
    """
    Reading old papers and feeling nostalgic? Fear not! This will change the
    default matplotlib settings to transport you right back several decades.
    """
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["font.family"] = "Routed Gothic"
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Routed Gothic"
    mpl.rcParams["mathtext.it"] = "Routed Gothic:italic"
    mpl.rcParams["mathtext.bf"] = "Routed Gothic:bold"
    mpl.rcParams["axes.titlesize"] = "x-large"


def stix():
    """
    Use STIX fonts
    """
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["font.family"] = "STIX Two Text"
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "STIX Two Math"
    mpl.rcParams["mathtext.it"] = "STIX Two Math:italic"
    mpl.rcParams["mathtext.bf"] = "STIX Two Math:bold"
    mpl.rcParams["axes.titlesize"] = "large"


def back2future():
    """
    Activate matplotlib settings from the default matplotlibrc file.
    """
    print("Activating settings from", mpl.matplotlib_fname())
    mpl.rc_file_defaults()
    mpl.rcParams["axes.titlesize"] = "medium"


def switch_backend():
    """
    Use to switch between regular inline and ipympl backend.
    """
    backend_list = [
        "module://ipykernel.pylab.backend_inline",
        "module://ipympl.backend_nbagg",
    ]
    current_backend = mpl.get_backend()
    if current_backend == backend_list[0]:
        ipython.magic("matplotlib ipympl")
        print("switched to ipympl plots")
    else:
        ipython.magic("matplotlib inline")
        print("switched to inline plots")


def quickfig(fs=10, yi=True, w=6, h=4, fgs=None):
    """
    Quick single pane figure.
    
    Automatically sets yaxis to be decreasing upwards so 
    we can plot against depth.
    
    Also closes all other figures for convenience.

    Parameters
    ----------
    fs : int, optional
        Fontsize (default 10)
    yi : bool, optional
        Increasing yaxis (default False)
    w : float, optional
        Figure width in inches (default 6)
    h : float, optional
        Figure height in inches (default 4)
    fgs : (float, float)
        Figure size, constructed as (w, h) if not specified here.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis handle
    """
    if fgs is None:
        fgs = (w, h)

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=fgs, constrained_layout=True, dpi=75
    )
    axstyle(ax, fontsize=fs)
    if yi is False:
        ax.invert_yaxis()
    ax.autoscale()

    # some adjustments when using ipympl
    current_backend = mpl.get_backend()
    if current_backend == "module://ipympl.backend_nbagg":
        fig.canvas.header_visible = False
        fig.canvas.toolbar_position = "bottom"
        fig.canvas.resizable = False

    return fig, ax


def newfig(width=7.5, height=5.5, fontsize=12):
    """
    Create new figure with own style.

    Parameters
    ----------
    width : float (optional)
        Figure width in inch
    height : float (optional)
        Figure height in inch
    fontsize : int (optional)
        Fontsize for tick labels, axis labels

    Returns
    -------
    fig : Figure handle
    ax : Axis handle
    """

    fig = plt.figure(figsize=(width, height))
    ax = plt.subplot(111)

    # Get rid of ticks. The position of the numbers is informative enough of
    # the position of the value.
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # Remove top and right axes lines ("spines")
    spines_to_remove = ["top", "right"]
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    # For remaining spines, thin out their line and change
    # the black to a slightly off-black dark grey
    almost_black = "#262626"
    spines_to_keep = ["bottom", "left"]
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(almost_black)
        ax.spines[spine].set_position(("outward", 5))

    # Change the labels to the off-black and adjust fontsize etc.
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
        length=0,
        colors=almost_black,
        direction="in",
    )
    ax.yaxis.label.set_size(fontsize)
    ax.xaxis.label.set_size(fontsize)

    # Change the axis title to off-black
    ax.title.set_color(almost_black)

    # turn grid on
    ax.grid(
        b=True,
        which="major",
        axis="both",
        color="0.7",
        linewidth=0.75,
        linestyle="-",
        alpha=0.8,
    )

    # Change figure position on screen
    # plt.get_current_fig_manager().window.setGeometry(0,0,width,height)

    return fig, ax


def axstyle(ax=None, fontsize=12, nospine=False, grid=True, ticks="off"):
    """
    Apply own style to axis.

    Parameters
    ----------
    ax : AxesSubplot (optional)
        Current axis will be chosen if no axis provided

    Returns
    -------
    ax : AxesSubplot
        Axis handle
    """

    if ax is None:
        ax = plt.gca()

    # Remove top and right axes lines ("spines")
    spines_to_remove = ["top", "right"]
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    # Remove bottom and left spines as well if desired
    if nospine:
        more_spines_to_remove = ["bottom", "left"]
        for spine in more_spines_to_remove:
            ax.spines[spine].set_visible(False)

    if ticks == "off":
        # Get rid of ticks.
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
    elif ticks == "in":
        ax.tick_params(axis="both", direction="in", length=2)

    # For remaining spines, thin out their line and change
    # the black to a slightly off-black dark grey
    almost_black = "#262626"
    spines_to_keep = ["bottom", "left"]
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(almost_black)
        ax.spines[spine].set_position(("outward", 5))

    # Change the labels to the off-black
    ax.xaxis.label.set_color(almost_black)
    ax.yaxis.label.set_color(almost_black)
    ax.yaxis.label.set_size(fontsize)
    ax.yaxis.offsetText.set_fontsize(fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.xaxis.offsetText.set_fontsize(fontsize)

    # Change the labels to the off-black
    ax.tick_params(
        axis="both", which="major", labelsize=fontsize, colors=almost_black,
    )

    # Change the axis title to off-black
    ax.title.set_color(almost_black)
    ax.title.set_size(fontsize + 1)

    # turn grid on
    if grid:
        ax.grid(
            b=True,
            which="major",
            axis="both",
            color="0.5",
            linewidth=0.25,
            linestyle="-",
            alpha=0.8,
        )

    # change legend fontsize (if there is one)
    try:
        plt.setp(ax.get_legend().get_texts(), fontsize=fontsize)
    except AttributeError:
        noleg = 1

    return ax


def newfigyy(width=7.5, height=5.5, fontsize=12):
    """Create figure with own style. Two y-axes.

    Set up figure with floating axes by defining `width` and `height`.
    Based on newfig.

    Parameters
    ----------
    width : float (optional)
        Figure width in inch
    height : float (optional)
        Figure height in inch
    fontsize : float (optional)
        Fontsize for tick labels, axis labels

    Returns
    -------
    fig : Figure handle
    ax1, ax2 : Axis handles

    """

    fig, ax1 = newfig(width, height)
    ax2 = ax1.twinx()
    ax1 = axstyle(ax1)
    spines_to_remove = ["top", "left", "bottom"]
    for spine in spines_to_remove:
        ax2.spines[spine].set_visible(False)
    ax2.xaxis.set_ticks_position("none")
    ax2.yaxis.set_ticks_position("none")
    almost_black = "#262626"
    spines_to_keep = ["right"]
    for spine in spines_to_keep:
        ax2.spines[spine].set_linewidth(0.5)
        ax2.spines[spine].set_color(almost_black)
        ax2.spines[spine].set_position(("outward", 5))
    ax2.xaxis.label.set_color(almost_black)
    ax2.yaxis.label.set_color(almost_black)
    return fig, ax1, ax2


def vstep(x, y, ax=None, *args, **kwargs):
    """
    Plot vertical steps.

    Parameters
    ----------
    x : array-like
        1-D sequence of x positions
    y : array-like
        1-D sequence of y positions. It is assumed, but not checked, that it is uniformly increasing.

    Returns
    -------
    lines : list
        List of `matplotlib.lines.Line2D` objects representing the plotted data.
    """
    if ax is None:
        ax = plt.gca()
    dy = np.diff(y)
    dy1 = np.insert(dy, 0, dy[0])
    dy2 = np.append(dy, dy[-1])
    y1 = y - dy1 / 2
    y2 = y + dy2 / 2
    Y = np.vstack([y1, y2]).transpose().flatten()
    X = np.vstack([x, x]).transpose().flatten()
    lines = ax.plot(X, Y, *args, **kwargs)
    return lines


def pcm(*args, **kwargs):
    """
    Wrapper for matplotlib's pcolormesh, blanking out nan's and
    thereby getting the auto-range right on arrays that include nan's.

    Parameters
    ----------
    x, y : float
        coordinates in x and y (optional)
    z : numpy array
        Data array
    Returns
    -------
    h : Plot handle

    Partly based on xarray code.

    """

    if len(args) == 1:
        z = args[0]
    elif len(args) == 3:
        x, y, z = args

    # set vmin, vmax based on percentiles and determine whether this is a
    # diverging ataset or not
    calc_data = np.ravel(z)
    calc_data = calc_data[np.isfinite(calc_data)]
    vmin = np.percentile(calc_data, 2.0)
    vmax = np.percentile(calc_data, 100.0 - 2.0)
    if (vmin < 0) and (vmax > 0):
        diverging = True
        center = 0
        vlim = max(abs(vmin - center), abs(vmax - center))
        vmin, vmax = -vlim, vlim
        vmin += center
        vmax += center
    else:
        diverging = False

    if "cmap" not in kwargs:
        if diverging:
            kwargs["cmap"] = "RdBu_r"
        else:
            kwargs["cmap"] = "Spectral_r"

    if len(args) == 1:
        if "ax" in kwargs:
            pax = kwargs["ax"]
            del kwargs["ax"]
            h = pax.pcolormesh(np.ma.masked_invalid(z), vmin=vmin, vmax=vmax, **kwargs)
        else:
            h = plt.pcolormesh(np.ma.masked_invalid(z), vmin=vmin, vmax=vmax, **kwargs)

    elif len(args) == 3:
        if "ax" in kwargs:
            pax = kwargs["ax"]
            del kwargs["ax"]
            h = pax.pcolormesh(
                x, y, np.ma.masked_invalid(z), vmin=vmin, vmax=vmax, **kwargs
            )
        else:
            h = plt.pcolormesh(
                x, y, np.ma.masked_invalid(z), vmin=vmin, vmax=vmax, **kwargs
            )
    else:
        print("You need to pass either 1 (z) or 3 (x,y,z) arguments.")

    return h


def png(fname, figdir="fig", dpi=300):
    """
    Save figure as png.

    Parameters
    ----------
    fname : str
        Figure name without file extension.

    figdir : str or Path object
        Path to figure directory. Default ./fig/

    dpi : int
        Resolution (default 200)
    """
    # get current working directory
    cwd = Path.cwd()
    # see if we already have a figure directory
    savedir = cwd.joinpath(figdir)
    if savedir.exists() and savedir.is_dir():
        print("saving to {}/".format(figdir))
    else:
        print("creating figure directory at {}/".format(savedir))
        savedir.mkdir()
    fname = fname + ".png"
    plt.savefig(savedir.joinpath(fname), dpi=dpi, bbox_inches="tight")


def figsave(fname, dirname="fig"):
    """
    adapted from https://github.com/jklymak/pythonlib/jmkfigure.py
    provide filename (fname)
    """
    import os

    try:
        os.mkdir(dirname)
    except:
        pass

    if dirname == "fig":
        pwd = os.getcwd() + "/fig/"
    else:
        pwd = dirname + "/"
    plt.savefig(dirname + "/" + fname + ".pdf", dpi=150, bbox_inches="tight")
    plt.savefig(dirname + "/" + fname + ".png", dpi=200, bbox_inches="tight")

    fout = open(dirname + "/" + fname + ".tex", "w")
    str = """\\begin{{figure*}}[htbp]
\\centering
\\includegraphics[width=1.0\\textwidth]{{{fname}}}
\\caption{{  \\newline \\hspace{{\\linewidth}}   {{\\footnotesize {pwd}{fname}.pdf}}}}
\\label{{fig:{fname}}}
\\end{{figure*}}""".format(
        pwd=pwd, fname=fname
    )
    fout.write(str)
    fout.close()

    cmd = "less " + dirname + "/%s.tex | pbcopy" % fname
    os.system(cmd)
    print("figure printed to {}".format(pwd))


def quickbasemap(ax, lon, lat, field=None):
    """
    Plot a quick map using basemap.

    Parameters
    ----------
    ax : axis object
        Handle to axis
    lon, lat : float
        Longitude / Latitude
    field : float
        Field to plot on map

    Returns
    -------
    m : basemp object
        handle to the map
    x, y : float
        lon, lat in map coordinates for plotting
    """
    from mpl_toolkits.basemap import Basemap

    m = Basemap(
        llcrnrlon=np.min(lon),
        llcrnrlat=np.min(lat),
        urcrnrlon=np.max(lon),
        urcrnrlat=np.max(lat),
        resolution="l",
        area_thresh=1000.0,
        projection="gall",
        lat_0=np.max(lat) - np.min(lat),
        lon_0=np.max(lon) - np.min(lon),
        ax=ax,
    )
    lonm, latm = np.meshgrid(lon, lat)
    x, y = m(lonm, latm)
    if field is not None:
        m.contourf(x, y, field, ax=ax)
    return m, x, y


def add_cax(fig, width=0.01, pad=0.01):
    """
    Add a colorbar axis to a row of axes (after last axis.) This axis can then 
    be passed on to a colorbar call.

    Parameters
    ----------
    fig : figure handle
        Handle to figure.
    width : float
        Width of colorbar (optional, defaults to 0.01)
    pad : float
        Padding between last axis and cax (optional, defaults to 0.01)

    Returns
    -------
    cax : AxesSubplot instance
    """
    ax = fig.axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax[-1])
    dpos = divider.get_position()
    cax = fig.add_axes([dpos[0] + dpos[2] + pad, dpos[1], width, dpos[3]])
    return cax


def ydecrease(ax=None):
    """
    Set decreasing yaxis as often desired when plotting a quantity
    against pressure or depth.

    Parameters
    ----------
    ax : axis handle
        Handle to axis (optional).
    """
    if ax is None:
        ax = plt.gca()
    ylims = ax.get_ylim()
    ax.set_ylim(bottom=np.amax(ylims), top=np.amin(ylims))


def ysym(ax=None):
    """
    Set ylim symmetric around zero based on current axis limits

    Parameters
    ----------
    ax : axis handle
        Handle to axis (optional).
    """
    if ax is None:
        ax = plt.gca()
    ylims = ax.get_ylim()
    absmax = np.max(np.abs(ylims))
    ax.set_ylim([-absmax, absmax])


def xsym(ax=None):
    """
    Set xlim symmetric around zero based on current axis limits

    Parameters
    ----------
    ax : axis handle
        Handle to axis.
    """
    if ax is None:
        ax = plt.gca()
    xlims = ax.get_xlim()
    absmax = np.max(np.abs(xlims))
    ax.set_xlim([-absmax, absmax])


def colcyc10(ax=None):
    """
    Set automatic color cycling for ax or current axis.

    Parameters
    ----------
    ax : axis handle
        Handle to axis.
    """
    from cycler import cycler

    if ax is None:
        ax = plt.gca()
    colors = [
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#CFECF9",
        "#7F7F7F",
        "#BCBD22",
        "#17BECF",
    ]
    ax.set_prop_cycle(cycler(color=colors))


def xytickdist(ax=None, x=1, y=1):
    """
    Set distance between ticks for xaxis and yaxis

    Parameters
    ----------
    ax : axis handle
        Handle to axis (optional).
    x : float
        Distance between xticks (default 1).
    y : float
        Distance between yticks (default 1).
    """
    if ax is None:
        ax = plt.gca()
    locx = mticker.MultipleLocator(base=x)
    ax.xaxis.set_major_locator(locx)
    locy = mticker.MultipleLocator(base=y)
    ax.yaxis.set_major_locator(locy)


def concise_date(ax=None, minticks=3, maxticks=10, show_offset=True, **kwargs):
    """
    Better date ticks using matplotlib's ConciseDateFormatter.

    Parameters
    ----------
    ax : axis handle
        Handle to axis (optional).
    minticks : int
        Minimum number of ticks (optional, default 6).
    maxticks : int
        Maximum number of ticks (optional, default 10).
    show_offset : bool, optional
        Show offset string to the right (default True).
        
    Note
    ----
    Currently only works for x-axis
    
    See Also
    --------
    matplotlib.mdates.ConciseDateFormatter : For formatting options that
      can be used here.
    """
    if ax is None:
        ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=show_offset, **kwargs)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def concise_date_all():
    import matplotlib.units as munits

    converter = mdates.ConciseDateConverter()
    munits.registry[np.datetime64] = converter


def cartopy_axes(ax, maxticks="auto"):
    """Requires cartopy."""
    if not _has_cartopy:
        raise ImportError("cartopy is required to do this.")
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="-",
    )
    gl.xlabels_top = False
    gl.ylabels_right = False
    if maxticks == "auto":
        gl.xlocator = mticker.AutoLocator()
        gl.ylocator = mticker.AutoLocator()
    else:
        gl.xlocator = mticker.MaxNLocator(maxticks)
        gl.ylocator = mticker.MaxNLocator(maxticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def multi_line(x, y, z, ax, **kwargs):
    """
    Plot multiple lines with color mapping.

    Parameters
    ----------
    x : array-like
        x-vector
    y : array-like
        Data mapped to color
    z : array-like
        Data
    ax : axis
        Axis
    Returns
    -------
    line : mpl.linecollection.LineCollection
        Lines
    """
    # see also here: https://matplotlib.org/examples/pylab_examples/multicolored_line.html
    norm = plt.Normalize(y.min(), y.max())
    segments = []
    zz = []
    for i, zi in enumerate(y):
        points = np.array([x, z[i, :]]).transpose()
        segments.append(points)
        zz.append(zi)
    segments = np.array(segments)
    lc = LineCollection(segments, cmap="Greys", norm=norm)
    lc.set_array(np.array(zz))
    lc.set_linewidth(1)
    # lc.set_alpha(0.5)
    line = ax.add_collection(lc)
    ax.autoscale()
    return line


def annotate_upper_left(text, ax):
    return ax.annotate(text, (0.02, 0.9), xycoords="axes fraction")


def cmap_partial(cmap_name, min, max):
    """
    Extract part of a colormap.

    Parameters
    ----------
    cmap_name : str
        Colormap name
    min : float
        Minimum in the range [0, 1]
    max : float
        Maximum in the range [0, 1]

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Colormap
    """
    interval = np.linspace(min, max)
    tmp = plt.cm.get_cmap(cmap_name)
    colors = tmp(interval)
    cmap = LinearSegmentedColormap.from_list("name", colors)
    return cmap
