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
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import string

from IPython import get_ipython

# import cm to register colormaps defined therein
from . import cm


def nostalgic():
    """
    Reading old papers and feeling nostalgic? Fear not! This will change the
    default matplotlib settings to transport you right back several decades.

    Notes
    -----
    Depends on Routed Gothic Font:
    https://webonastick.com/fonts/routed-gothic/
    """
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["font.family"] = "Routed Gothic"
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Routed Gothic"
    mpl.rcParams["mathtext.it"] = "Routed Gothic:italic"
    mpl.rcParams["mathtext.bf"] = "Routed Gothic:bold"
    mpl.rcParams["axes.titlesize"] = "x-large"


def helvetica():
    """
    Use Helvetica font for plotting.
    """
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["font.family"] = "Helvetica"
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Helvetica"
    mpl.rcParams["mathtext.it"] = "Helvetica:italic"
    mpl.rcParams["mathtext.bf"] = "Helvetica:bold"
    mpl.rcParams["axes.titlesize"] = "medium"
    mpl.rcParams["legend.fontsize"] = "small"


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
    ipython = get_ipython()
    backend_list = [
        "module://matplotlib_inline.backend_inline",
        "module://ipympl.backend_nbagg",
    ]
    current_backend = mpl.get_backend()
    if current_backend == backend_list[0]:
        ipython.magic("matplotlib widget")
        print("switched to ipympl plots")
    else:
        ipython.magic("matplotlib inline")
        print("switched to inline plots")


def quickfig(fs=10, yi=True, w=6, h=4, fgs=None, r=1, c=1, grid=False, **kwargs):
    """
    Quick single pane figure.

    Automatically sets yaxis to be decreasing upwards so
    we can plot against depth.

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
    r : int, optional
        Number of rows (default 1)
    c : int, optional
        Number of columns (default 1)
    grid : bool
        Show grid (default False)

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
        nrows=r,
        ncols=c,
        figsize=fgs,
        constrained_layout=True,
        dpi=75,
        **kwargs,
    )
    if isinstance(ax, np.ndarray):
        [axstyle(axi, fontsize=fs, grid=grid) for axi in ax.flatten()]
    else:
        axstyle(ax, fontsize=fs, grid=grid)
    if yi is False:
        ax.invert_yaxis()
    if r == 1 and c == 1:
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
        # b=True,
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


def axstyle(
    ax=None,
    fontsize=12,
    nospine=False,
    grid=True,
    ticks="off",
    ticklength=2,
    spine_offset=5,
):
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
    # find out background color - if this is set to ayu dark, adjust some axis
    # colors
    figcolor = plt.rcParams["figure.facecolor"]
    dark = True if figcolor == "#0d1318" else False

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

    # For remaining spines, thin out their line and change
    # the black to a slightly off-black dark grey
    almost_black = "#262626"
    # if figure background is dark, set this close to white
    if dark:
        almost_black = "#ebe6d7"

    if ticks == "off":
        # Change the labels to the off-black
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=fontsize,
            colors=almost_black,
        )
        # Get rid of ticks.
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
    elif ticks == "in":
        # Change the labels to the off-black
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=fontsize,
            colors=almost_black,
            direction="in",
            length=ticklength,
        )

    spines_to_keep = ["bottom", "left"]
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(almost_black)
        ax.spines[spine].set_position(("outward", spine_offset))

    # Change the labels to the off-black
    ax.yaxis.label.set_color(almost_black)
    ax.yaxis.label.set_size(fontsize)
    ax.yaxis.offsetText.set_fontsize(fontsize)
    ax.xaxis.label.set_color(almost_black)
    ax.xaxis.label.set_size(fontsize)
    ax.xaxis.offsetText.set_fontsize(fontsize)

    # Change the axis title to off-black
    ax.title.set_color(almost_black)
    ax.title.set_size(fontsize + 1)

    # turn grid on
    if grid:
        ax.grid(
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
        _noleg = 1

    return ax


def gridstyle(ax, which="major"):
    if which == "both":
        ax.grid(
            which="major",
            axis="both",
            color="0.4",
            linewidth=0.25,
            linestyle="-",
            alpha=0.8,
        )
        ax.grid(
            which="minor",
            axis="both",
            color="0.7",
            linewidth=0.25,
            linestyle="-",
            alpha=0.8,
        )
        ax.minorticks_on()
    else:
        ax.grid(
            which=which,
            axis="both",
            color="0.5",
            linewidth=0.25,
            linestyle="-",
            alpha=0.5,
        )


def tickstyle(ax, which="both", direction="in"):
    ax.tick_params(
        axis="both",
        which=which,
        direction=direction,
    )


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
    fig
        Figure handle
    ax1
        Axis handle
    ax2
        Axis handle
    """

    fig, ax1 = newfig(width, height, fontsize=fontsize)
    ax2 = ax1.twinx()
    ax1 = axstyle(ax1, fontsize=fontsize)
    spines_to_remove = ["top", "left", "bottom"]
    for spine in spines_to_remove:
        ax2.spines[spine].set_visible(False)
    ax2.xaxis.set_ticks_position("none")
    ax2.yaxis.set_ticks_position("none")
    almost_black = "#262626"
    spines_to_keep = ["right"]
    for spine in spines_to_keep:
        ax2.spines[spine].set_linewidth(1.0)
        ax2.spines[spine].set_color(almost_black)
        ax2.spines[spine].set_position(("outward", 5))
    ax2.xaxis.label.set_color(almost_black)
    ax2.yaxis.label.set_color(almost_black)
    ax2.yaxis.label.set_size(fontsize)
    ax2.yaxis.offsetText.set_fontsize(fontsize)
    ax2.xaxis.label.set_size(fontsize)
    ax2.xaxis.offsetText.set_fontsize(fontsize)
    ax2.tick_params(labelsize=fontsize)

    ax1.title.set_fontsize(fontsize + 1)
    ax2.title.set_fontsize(fontsize + 1)

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


def stickplot(ax, times, data, uv=True, units="", scale=0.1, color="k"):
    """
    Create a stick plot of the given data on the given axes.

    Stick plots are commonly used to display time series of
    a vector quantity at a point, such as wind or ocean current observations.

    Parameters
    ----------
    axis: matplotlib.axis.Axis
        The axis object to plot on.
    times: array-like
        Time vector
    data : tuple
        Input data as a tuple of either (u, v) or (speed, direction).
        Directions are in degrees from North, where North is up on the plot)
    uv : bool, optional
        Indicates whether input data are (u, v) or (speed, direction).
    units: str, optional
        Units of the observation. Defaults to empty string.
    scale: float, optional
        Data scale factor. Defaults to 0.1.

    Credits
    -------
    Chris Barker
    http://matplotlib.1069221.n5.nabble.com/Stick-Plot-td21479.html
    """

    props = {
        "units": "dots",
        "width": 2,
        "headwidth": 0,
        "headlength": 0,
        "headaxislength": 0,
        "scale": scale,
        "color": color,
    }

    # fixme: this should use some smarts to fit the data
    label_scale = 0.1
    unit_label = "%3g %s" % (label_scale, units)

    if uv:
        u, v = data
    else:
        directions, speeds = data
        dir_rad = directions / 180.0 * np.pi
        u = np.sin(dir_rad) * speeds
        v = np.cos(dir_rad) * speeds
    y = np.zeros_like(u)

    Q = ax.quiver(times, y, u, v, **props)
    ax.quiverkey(
        Q,
        X=0.1,
        Y=0.95,
        U=label_scale,
        label=unit_label,
        coordinates="axes",
        labelpos="S",
    )
    yaxis = ax.yaxis
    yaxis.set_ticklabels([])


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


def png(fname, figdir="fig", dpi=300, verbose=True, transparent=False):
    """
    Save figure to png file.

    Parameters
    ----------
    fname : str or Path
        Figure name if str, figure name and optionally absolute path as well if
        Path.
    figdir : str or Path, optional
        Path to figure directory. Defaults to ./fig; will be created if it does
        not exist.
    dpi : int, optional
        Resolution (default 300)
    verbose : bool, optional
        Print output path that the figure is saved to in screen.
    transparent : bool, optional
        Transparent figure background. Defaults to False.
    """
    savedir, name = _figure_name(fname, figdir, extension="png", verbose=verbose)
    if transparent:
        plt.savefig(
            savedir.joinpath(fname),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="none",
            edgecolor="none",
        )
    else:
        plt.savefig(
            savedir.joinpath(fname),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="w",
            edgecolor="none",
        )


def pdf(fname, figdir="fig", dpi=300, verbose=True, transparent=False):
    """
    Save figure to pdf file.

    Parameters
    ----------
    fname : str or Path
        Figure name if str, figure name and optionally absolute path as well if
        Path.
    figdir : str or Path, optional
        Path to figure directory. Defaults to ./fig; will be created if it does
        not exist.
    dpi : int, optional
        Resolution (default 300)
    verbose : bool, optional
        Print output path that the figure is saved to in screen.
    transparent : bool, optional
        Transparent figure background. Defaults to False.
    """
    savedir, name = _figure_name(fname, figdir, extension="pdf", verbose=verbose)
    if transparent:
        plt.savefig(
            savedir.joinpath(name),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="none",
            edgecolor="none",
        )
    else:
        plt.savefig(
            savedir.joinpath(name),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="w",
            edgecolor="none",
        )


def _figure_name(fname, figdir, extension, verbose=True):
    """
    Generate figure name/path for png and pdf functions.

    Parameters
    ----------
    fname : str or Path
        Figure name if str, figure name and optionally absolute path as well if
        Path.
    figdir : str or Path, optional
        Path to figure directory. Defaults to ./fig; will be created if it does
        not exist.
    extension : str
        png or pdf
    verbose : bool
        Print savedir to screen.

    Returns
    -------
    savedir : pathlib.Path
        Diretory.
    name : str
        File name.
    """
    # get current working directory
    cwd = Path.cwd()
    # see if we have a path instance with an absolute path. then we make this
    # the figure directory and file name.
    if isinstance(fname, Path):
        if fname.is_absolute():
            savedir = fname.parent
        else:
            savedir = cwd.joinpath(figdir)
        name = fname.stem
    # otherwise deal with str
    elif isinstance(fname, str):
        tmpname = Path(fname)
        if tmpname.is_absolute():
            savedir = tmpname.parent
        else:
            savedir = cwd.joinpath(figdir)
        name = tmpname.stem
    else:
        raise TypeError("Input must be str or Path instance")

    # see if we already have a figure directory
    if savedir.exists() and savedir.is_dir():
        if verbose:
            print("saving to {}/".format(savedir))
    else:
        if verbose:
            print("creating figure directory at {}/".format(savedir))
        savedir.mkdir()

    if extension[0] == ".":
        extension = extension[1:]
    name = name + "." + extension
    return savedir, name


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
\\end{{figure*}}""".format(pwd=pwd, fname=fname)
    fout.write(str)
    fout.close()

    cmd = "less " + dirname + "/%s.tex | pbcopy" % fname
    os.system(cmd)
    print("figure printed to {}".format(pwd))


def contourf_hide_edges(h):
    for c in h.collections:
        c.set_rasterized(True)
        c.set_edgecolor("face")


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


def find_cax():
    fig = plt.gcf()
    axs = fig.get_axes()
    cax = []
    for axi in axs:
        if "colorbar" in axi.axes.get_label():
            cax.append(axi)
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


def cycle_cmap(n=10, cmap="viridis", ax=None):
    """
    Set automatic color cycling through colormap for ax or current axis.

    Parameters
    ----------
    n : int, optional
        Number of colors. Defaults to 10.
    cmap : str, optional
        Colormap name. Defaults to 'viridis'.
    ax : axis handle, optional
        Handle to axis. Defaults to plt.gca().
    """

    if ax is None:
        ax = plt.gca()
    colors = [plt.get_cmap(cmap)(1.0 * i / n) for i in range(n)]
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
    # remove axis label "time" if present
    if ax.get_xlabel() == "time":
        _ = ax.set_xlabel("")


def concise_date_all():
    import matplotlib.units as munits

    converter = mdates.ConciseDateConverter()
    munits.registry[np.datetime64] = converter


def cartopy_axes(ax, maxticks="auto"):
    """Requires cartopy."""
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="-",
    )
    gl.top_labels = False
    gl.right_labels = False
    if maxticks == "auto":
        gl.xlocator = mticker.AutoLocator()
        gl.ylocator = mticker.AutoLocator()
    else:
        gl.xlocator = mticker.MaxNLocator(maxticks)
        gl.ylocator = mticker.MaxNLocator(maxticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def vlines(x, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    ax.vlines(x, ymin, ymax, **kwargs)


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


def annotate_corner(
    text,
    ax,
    quadrant=1,
    fw="bold",
    fs=10,
    addx=0,
    addy=0,
    col="k",
    background_circle=False,
    text_bg=None,
    text_bg_alpha=0.5,
):
    """Add text to axis corner.

    Parameters
    ----------
    text : str
        Text to add
    ax : matplotlib.axis
        Axis
    quadrant : int [1, 2, 3, 4]
        Corner
    fw : str, optional
        Font weight
    fs : int, optional
        Font size
    addx : float, optional
        Position offset in x
    addy : float, optional
        Position offset in y
    col : str, optional
        Text color
    background_circle : bool or str, optional
        Draw background circle behind text.
        Can be a string defining circle color.
        Default color is white.
        Defaults to False.
    text_bg : str, optional
        Text background color. Defaults to None.
    text_bg_alpha : float
        Alpha for text_bg. Default 0.5.

    Returns
    -------
    h : matplotlib.Annotation

    """

    if background_circle is True:
        background_circle = "w"
    if quadrant == 1:
        loc = (0.02 + addx, 0.9 + addy)
        ha = "left"
    if quadrant == 2:
        loc = (0.02 + addx, 0.1 + addy)
        ha = "left"
    if quadrant == 3:
        loc = (0.98 + addx, 0.1 + addy)
        ha = "right"
    if quadrant == 4:
        loc = (0.98 + addx, 0.9 + addy)
        ha = "right"
    if background_circle:
        h = ax.annotate(
            text,
            loc,
            xycoords="axes fraction",
            fontweight=fw,
            fontsize=fs,
            color=col,
            ha=ha,
            backgroundcolor="w",
            bbox=dict(
                boxstyle="circle",
                edgecolor=background_circle,
                facecolor=background_circle,
            ),
        )
    elif text_bg:
        h = ax.annotate(
            text,
            loc,
            xycoords="axes fraction",
            fontweight=fw,
            fontsize=fs,
            color=col,
            ha=ha,
            bbox=dict(
                edgecolor="none",
                facecolor=text_bg,
                alpha=text_bg_alpha,
            ),
        )
    else:
        h = ax.annotate(
            text,
            loc,
            xycoords="axes fraction",
            fontweight=fw,
            fontsize=fs,
            color=col,
            ha=ha,
        )
    return h


def subplotlabel(ax, color="k", fs=10, fw="bold", bg="w", bga=1, x=0, y=0.96):
    """Add alphabetic subplot labels to an array of axes.

    Parameters
    ----------
    ax : np.array
        Array with axis instances.
    color : str, optional
        Font color. Default black.
    fs : int, optional
        Font size. Default 10.
    fw : str, optional
        Font weight. Default bold.
    bg : str, optional
        Background color
    bga : float [0...1], optional
        Background alpha (transparency). Default 1 (not transparent).
    x : float, optional
        x-position (in axis units). Default 0.
    y : float, optional
        y-position (in axis units). Default 0.96.

    Returns
    -------
    list
        List of matplotlib.Annotation objects.
    """
    out = []
    atoz = string.ascii_lowercase
    n = len(ax.flatten())
    sublabelspecs = dict(
        xycoords="axes fraction",
        color=color,
        fontweight=fw,
        fontsize=fs,
        bbox=dict(facecolor=bg, edgecolor="none", alpha=bga, boxstyle="circle,pad=0.1"),
    )
    for axi, letter in zip(ax.flatten(), atoz[:n]):
        out.append(axi.annotate(letter, (x, y), **sublabelspecs))


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


def get_max_zorder(ax):
    """Get highest zorder of all elements in an Axis instance.

    Parameters
    ----------
    ax : matplotlib.axes.Axis
        Axis

    Returns
    -------
    int
        Maximum zorder
    """
    return max([_.zorder for _ in ax.get_children()])


def remove_axis_labels(ax):
    if isinstance(ax, np.ndarray):
        [axi.set(xlabel="", ylabel="", title="") for axi in ax]
    else:
        ax.set(xlabel="", ylabel="")
    return
