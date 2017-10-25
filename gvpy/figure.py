#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvpy.figure for matplotlib related stuff.

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def newfig(width=7.5, height=5.5):
    """Create figure with own style.

    Set up figure with floating axes by defining `width` and `height`.
    Based on gv1 style.

    Parameters
    ----------
    width : float
        Figure width in inch
    height : float
        Figure height in inch

    Returns
    -------
    fig : Figure handle
    ax : Axis handle

    """
    # plt.rc('figure', figsize=(width, height))
    # plt.rc('font', size=12)
    # plt.rc('font',family='sans-serif')
    # mpl.rcParams['font.family'] = 'sans-serif'
    # mpl.rcParams['font.sans-serif'] = ['Helvetica']
    # mpl.rcParams['font.variant'] = 'normal'
    # mpl.rcParams['font.weight'] = 'normal'

    # mpl.rcParams['mathtext.fontset'] = 'custom'
    # mpl.rcParams['mathtext.rm'] = 'Helvetica'
    # mpl.rcParams['mathtext.it'] = 'Helvetica:italic'
    # mpl.rcParams['mathtext.bf'] = 'Helvetica:bold'

    with plt.style.context(('gv1')):
        fig = plt.figure(figsize=(width, height))
        ax = plt.subplot(111)

        # Get rid of ticks. The position of the numbers is informative enough of
        # the position of the value.
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        # For remaining spines, thin out their line and change
        # the black to a slightly off-black dark grey
        almost_black = '#262626'
        spines_to_keep = ['bottom', 'left']
        for spine in spines_to_keep:
            ax.spines[spine].set_linewidth(0.5)
            ax.spines[spine].set_color(almost_black)
            ax.spines[spine].set_position(('outward', 5))

        # Change the labels to the off-black
        ax.xaxis.label.set_color(almost_black)
        ax.yaxis.label.set_color(almost_black)

        # Change the axis title to off-black
        ax.title.set_color(almost_black)

        # Change figure position on screen
        # plt.get_current_fig_manager().window.setGeometry(0,0,width,height)

    return fig, ax


def newfigyy(width=7.5, height=5.5):
    """Create figure with own style. Two y-axes.

    Set up figure with floating axes by defining `width` and `height`.
    Based on newfig.

    Parameters
    ----------
    width : float
        Figure width in inch
    height : float
        Figure height in inch

    Returns
    -------
    fig : Figure handle
    ax1, ax2 : Axis handles

    """

    fig, ax1 = newfig(width, height)
    ax2 = ax1.twinx()
    ax1 = axstyle(ax1)
    spines_to_remove = ['top', 'left', 'bottom']
    for spine in spines_to_remove:
        ax2.spines[spine].set_visible(False)
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    almost_black = '#262626'
    spines_to_keep = ['right']
    for spine in spines_to_keep:
        ax2.spines[spine].set_linewidth(0.5)
        ax2.spines[spine].set_color(almost_black)
        ax2.spines[spine].set_position(('outward', 5))
    ax2.xaxis.label.set_color(almost_black)
    ax2.yaxis.label.set_color(almost_black)
    return fig, ax1, ax2


def axstyle(ax=plt.gca()):
    """
    ax = axstyle(ax)):
    Apply own style to axis.
    """
    plt.rc('font', size=10)
    plt.rc('font', family='sans-serif')
    mpl.rcParams['font.sans-serif'] = ['Helvetica']
    mpl.rcParams['font.variant'] = 'normal'
    mpl.rcParams['font.weight'] = 'normal'

    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Helvetica'
    mpl.rcParams['mathtext.it'] = 'Helvetica:italic'
    mpl.rcParams['mathtext.bf'] = 'Helvetica:bold'

    # AXES
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.linewidth'] = 0.5

    # TICKS
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    # GRID
    mpl.rcParams['grid.color'] = (0.7, 0.7, 0.7)
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['grid.linestyle'] = '-'
    mpl.rcParams['grid.alpha'] = 0.8

    # Remove top and right axes lines ("spines")
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    # Get rid of ticks. The position of the numbers is informative enough of
    # the position of the value.
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # For remaining spines, thin out their line and change
    # the black to a slightly off-black dark grey
    almost_black = '#262626'
    spines_to_keep = ['bottom', 'left']
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(almost_black)
        ax.spines[spine].set_position(('outward', 5))

    # Change the labels to the off-black
    ax.xaxis.label.set_color(almost_black)
    ax.yaxis.label.set_color(almost_black)

    # Change the axis title to off-black
    ax.title.set_color(almost_black)
    return ax


def pcm(*args, **kwargs):
    """
    Wrapper for matplotlib's pcolormesh, blanking out nan's and thereby getting
    the auto-range right on arrays that include nan's.

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

    # set vmin, vmax based on percentiles and determine whether this is a diverging
    # dataset or not
    calc_data = np.ravel(z)
    calc_data = calc_data[np.isfinite(calc_data)]
    vmin = np.percentile(calc_data, 2.0)
    vmax = np.percentile(calc_data, 100.0-2.0)
    if (vmin<0) and (vmax>0):
        diverging = True
        center = 0
        vlim = max(abs(vmin - center), abs(vmax - center))
        vmin, vmax = -vlim, vlim
        vmin += center
        vmax += center
    else:
        diverging = False

    if 'cmap' not in kwargs:
        if diverging:
            kwargs['cmap'] = 'RdBu_r'
        else:
            kwargs['cmap'] = 'Spectral_r'

    if len(args)==1:
        if 'ax' in kwargs:
            pax = kwargs['ax']
            del kwargs['ax']
            h = pax.pcolormesh(np.ma.masked_invalid(z),vmin=vmin,vmax=vmax,**kwargs)
        else:
            h = plt.pcolormesh(np.ma.masked_invalid(z),vmin=vmin,vmax=vmax,**kwargs)

    elif len(args)==3:
        if 'ax' in kwargs:
            pax = kwargs['ax']
            del kwargs['ax']
            h = pax.pcolormesh(x,y,np.ma.masked_invalid(z),vmin=vmin,vmax=vmax, **kwargs)
        else:
            h = plt.pcolormesh(x,y,np.ma.masked_invalid(z),vmin=vmin,vmax=vmax, **kwargs)
    else:
        print('You need to pass either 1 (z) or 3 (x,y,z) arguments.')

    return h

def figsave(fname, dirname='fig'):
    """
    adapted from https://github.com/jklymak/pythonlib/jmkfigure.py
    provide filename (fname)
    """
    import os

    try:
        os.mkdir(dirname)
    except:
        pass

    if dirname == 'fig':
        pwd = os.getcwd()+'/fig/'
    else:
        pwd = dirname+'/'
    plt.savefig(dirname+'/'+fname+'.pdf', dpi=150)
    plt.savefig(dirname+'/'+fname+'.png', dpi=150)

    fout = open(dirname+'/'+fname+'.tex', 'w')
    str = """\\begin{{figure*}}[htbp]
\\centering
\\includegraphics[width=1.0\\textwidth]{{{fname}}}
\\caption{{  \\newline \\hspace{{\\linewidth}}   {{\\footnotesize {pwd}{fname}.pdf}}}}
\\label{{fig:{fname}}}
\\end{{figure*}}""".format(pwd=pwd, fname=fname)
    fout.write(str)
    fout.close()

    cmd = 'less '+dirname+'/%s.tex | pbcopy' % fname
    os.system(cmd)
    print('figure printed to {}'.format(pwd))
