#!/usr/bin/env python
# Filename: gvfigure.py

import matplotlib.pyplot as plt
import matplotlib as mpl

def dgvfigure(width=7.5,height=5.5):
  """
  fig, ax = dgvfigure(width,height):
  Create figure with own style.  
  """
  
  plt.rc('figure',figsize=(width,height),dpi=300)
  plt.rc('font',size=9)
  plt.rc('font',family='sans-serif');
  # mpl.rcParams['font.family'] = 'sans-serif'
  mpl.rcParams['font.sans-serif'] = 'Helvetica'
  mpl.rcParams['font.variant'] = 'normal';
  mpl.rcParams['font.weight'] = 'normal';

  mpl.rcParams['mathtext.fontset'] = 'custom'
  mpl.rcParams['mathtext.rm'] = 'Helvetica'
  mpl.rcParams['mathtext.it'] = 'Helvetica:italic'
  mpl.rcParams['mathtext.bf'] = 'Helvetica:bold'
  
  # AXES
  mpl.rcParams['axes.grid'] = True;
  mpl.rcParams['axes.linewidth'] = 0.5;

  # TICKS
  mpl.rcParams['xtick.direction'] = 'in';
  mpl.rcParams['ytick.direction'] = 'in';
  
  # GRID
  mpl.rcParams['grid.color'] = (0.7,0.7,0.7);
  mpl.rcParams['grid.linewidth'] = 0.5;
  mpl.rcParams['grid.linestyle'] = '-';
  mpl.rcParams['grid.alpha'] = 0.8;

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

def axstyle(ax):
  """
  ax = axstyle(ax)):
  Apply own style to axis.
  """
  plt.rc('font',size=9)
  plt.rc('font',family='sans-serif');
  mpl.rcParams['font.sans-serif'] = 'Helvetica'
  mpl.rcParams['font.variant'] = 'normal';
  mpl.rcParams['font.weight'] = 'normal';

  mpl.rcParams['mathtext.fontset'] = 'custom'
  mpl.rcParams['mathtext.rm'] = 'Helvetica'
  mpl.rcParams['mathtext.it'] = 'Helvetica:italic'
  mpl.rcParams['mathtext.bf'] = 'Helvetica:bold'
  
  # AXES
  mpl.rcParams['axes.grid'] = True;
  mpl.rcParams['axes.linewidth'] = 0.5;

  # TICKS
  mpl.rcParams['xtick.direction'] = 'in';
  mpl.rcParams['ytick.direction'] = 'in';
  
  # GRID
  mpl.rcParams['grid.color'] = (0.7,0.7,0.7);
  mpl.rcParams['grid.linewidth'] = 0.5;
  mpl.rcParams['grid.linestyle'] = '-';
  mpl.rcParams['grid.alpha'] = 0.8;

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