import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import gvpy as gv


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_calc_E_omg_southern_hemisphere_matches_northern():
    """Regression test: negative f gave log10 of a negative number and NaN spectra."""
    north = gv.gm81.calc_E_omg(N=2e-3, lat=30)
    south = gv.gm81.calc_E_omg(N=2e-3, lat=-30)
    assert np.isfinite(south.KE).all() and np.isfinite(south.PE).all()
    np.testing.assert_allclose(south.omega, north.omega)
    np.testing.assert_allclose(south.KE, north.KE)
    np.testing.assert_allclose(south.PE, north.PE)


def test_plot_E_omg_southern_hemisphere_matches_northern():
    _, omg_n, K_n, P_n = gv.gm81.plot_E_omg(N=2e-3, lat=30)
    _, omg_s, K_s, P_s = gv.gm81.plot_E_omg(N=2e-3, lat=-30)
    assert np.isfinite(K_s).all()
    np.testing.assert_allclose(omg_s, omg_n)
    np.testing.assert_allclose(K_s, K_n)
    np.testing.assert_allclose(P_s, P_n)


def test_plot_spectrum_southern_hemisphere():
    """The inertial frequency markers and GM curve match the northern hemisphere.

    Regression test: f < 0 in the south placed the f and 2f markers at negative
    frequencies and made the GM spectrum NaN.
    """
    time = np.datetime64("2025-01-01") + np.arange(2000) * np.timedelta64(1, "h")
    da = xr.DataArray(
        np.random.default_rng(0).normal(size=time.size), coords=dict(time=time)
    )
    ticks = []
    for lat in [30.0, -30.0]:
        ax = da.gv.plot_spectrum(lat=lat, N=2e-3)
        ticks.append(ax.child_axes[0].get_xticks())
        gm = next(line for line in ax.get_lines() if line.get_label() == "KE")
        assert np.isfinite(gm.get_ydata()).all()
    np.testing.assert_allclose(ticks[1], ticks[0])
    assert (ticks[1] > 0).all()
