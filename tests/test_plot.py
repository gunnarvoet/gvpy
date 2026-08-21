import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import gvpy as gv


@pytest.fixture(autouse=True)
def close_figures():
    """Keep figures from accumulating across tests."""
    yield
    plt.close("all")


def axes_list(ax):
    """Return the axes of a quickfig result as a flat list."""
    return list(ax.flatten()) if isinstance(ax, np.ndarray) else [ax]


@pytest.mark.parametrize("r, c", [(1, 1), (1, 3), (2, 1), (2, 2)])
def test_quickfig_yi_false_inverts_every_axes(r, c):
    """yi=False inverts the y-axis for multi-panel figures, not just single ones.

    Regression test: the inversion used to be applied to the return value of
    ``plt.subplots`` without checking whether it was an array, so any figure
    with more than one panel raised AttributeError instead of inverting.
    """
    _fig, ax = gv.plot.quickfig(r=r, c=c, yi=False)
    axx = axes_list(ax)
    assert len(axx) == r * c
    assert all(axi.yaxis_inverted() for axi in axx)


@pytest.mark.parametrize("r, c", [(1, 1), (1, 3), (2, 2)])
def test_quickfig_default_leaves_y_increasing(r, c):
    """The default yi=True leaves the y-axis alone."""
    _fig, ax = gv.plot.quickfig(r=r, c=c)
    assert not any(axi.yaxis_inverted() for axi in axes_list(ax))
