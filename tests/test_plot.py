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


@pytest.mark.parametrize("fname", ["fig.v2", "abc2"])
def test_png_forces_png_extension(tmp_path, fname):
    """Regression test: png() saved to the raw fname, so the extension was not forced."""
    fig, ax = gv.plot.quickfig()
    gv.plot.png(fname, figdir=tmp_path, verbose=False)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".png"


def test_subplotlabel_returns_annotations():
    fig, ax = gv.plot.quickfig(r=2, c=2)
    out = gv.plot.subplotlabel(ax)
    assert [a.get_text() for a in out] == ["a", "b", "c", "d"]


def test_remove_axis_labels_2d_axes():
    """Regression test: a 2-D axes array raised AttributeError."""
    fig, ax = gv.plot.quickfig(r=2, c=2)
    for axi in ax.flatten():
        axi.set(xlabel="x", ylabel="y")
    gv.plot.remove_axis_labels(ax)
    assert all(axi.get_xlabel() == "" for axi in ax.flatten())


def test_pcm_wrong_number_of_args():
    with pytest.raises(TypeError):
        gv.plot.pcm(np.arange(3), np.ones((3, 3)))


def test_cmap_partial_no_deprecation_warning():
    """Regression test: plt.cm.get_cmap is deprecated (warnings fail tests)."""
    gv.plot.cmap_partial("viridis", 0.2, 0.8)
