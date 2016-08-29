#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvfigure for matplotlib related stuff.

"""

import matplotlib as mpl
import matplotlib.pyplot as plt


def newfig(width=7.5, height=5.5):
    """Create figure with own style.

    Set up figure with floating axes by defining `width` and `height`. This
    uses matplotlib's rcParams.

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
    plt.rc('figure', figsize=(width, height))
    plt.rc('font', size=10)
    # plt.rc('font',family='sans-serif')
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
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

    fig = plt.figure()
    ax = plt.subplot(111)

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

    return fig, ax


def newfigyy(width=7.5, height=5.5):
    """Create figure with own style. Two y-axes.

    Set up figure with floating axes by defining `width` and `height`. This
    uses matplotlib's rcParams.

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

    fig, ax1 = newfig(10, 4)
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


def axstyle(ax):
    """
    ax = axstyle(ax)):
    Apply own style to axis.
    """
    plt.rc('font', size=9)
    plt.rc('font', family='sans-serif')
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
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


def gvprint(fname, pyname, dirname='fig'):
    """
    adapted from https://github.com/jklymak/pythonlib/jmkfigure.py
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
    plt.savefig(dirname+'/'+fname+'.pdf', dpi=400)
    plt.savefig(dirname+'/'+fname+'.png', dpi=400)

    fout = open(dirname+'/'+fname+'.tex', 'w')
    str = """\\begin{{figure*}}[htbp]
\\centering
\\includegraphics[width=1.0\\textwidth]{{{fname}}}
\\caption{{  \\newline \\hspace{{\\linewidth}}   {{\\footnotesize {pwd}{fname}.pdf}}}}
\\label{{fig:{fname}}}
\\end{{figure*}}""".format(pwd=pwd, pyname=pyname, fname=fname)
    fout.write(str)
    fout.close()

    cmd = 'less '+dirname+'/%s.tex | pbcopy' % fname
    os.system(cmd)
