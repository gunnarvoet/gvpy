"""Tests for gvpy.signal."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

import gvpy as gv

FS = 1.0
N = 8192


def _series(seed=0, n=N):
    """Red noise, which is coherent with a shifted copy of itself."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))


def _shift(x, tau, fs=FS):
    """Shift x later in time by tau, in units of 1/fs, via an FFT phase ramp."""
    f = np.fft.rfftfreq(x.size, d=1 / fs)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * tau), n=x.size)


@pytest.mark.parametrize("tau", [-3.7, -1.0, -0.25, 0.0, 0.25, 1.0, 3.7])
def test_band_lag_recovers_imposed_shift(tau):
    x = _series()
    res = gv.signal.band_lag(x, _shift(x, tau), fs=FS, nperseg=1024, band=(0.005, 0.05))
    assert res.lag == pytest.approx(tau, abs=0.02)


def test_band_lag_sign_convention():
    """A positive lag means y happens later than x."""
    x = _series()
    res = gv.signal.band_lag(x, _shift(x, 2.0), fs=FS, nperseg=1024, band=(0.005, 0.05))
    assert res.lag > 0


def test_band_lag_is_antisymmetric():
    x = _series()
    y = _shift(x, 1.5)
    fwd = gv.signal.band_lag(x, y, fs=FS, nperseg=1024, band=(0.005, 0.05))
    rev = gv.signal.band_lag(y, x, fs=FS, nperseg=1024, band=(0.005, 0.05))
    assert fwd.lag == pytest.approx(-rev.lag, abs=1e-6)


def test_band_lag_independent_of_band_for_a_pure_shift():
    """The defining property: a time shift puts one slope on every band."""
    x = _series()
    y = _shift(x, 1.2)
    narrow = gv.signal.band_lag(x, y, fs=FS, nperseg=1024, band=(0.005, 0.02))
    wide = gv.signal.band_lag(x, y, fs=FS, nperseg=1024, band=(0.005, 0.1))
    assert narrow.lag == pytest.approx(wide.lag, abs=0.02)


def test_band_lag_scales_with_fs():
    """fs sets the units; the lag is in 1/fs."""
    x = _series()
    y = _shift(x, 2.0)
    a = gv.signal.band_lag(x, y, fs=1.0, nperseg=1024, band=(0.005, 0.05))
    b = gv.signal.band_lag(x, y, fs=10.0, nperseg=1024, band=(0.05, 0.5))
    assert a.lag == pytest.approx(10 * b.lag, rel=1e-6)


def test_band_lag_reports_diagnostics():
    x = _series()
    res = gv.signal.band_lag(x, _shift(x, 0.5), fs=FS, nperseg=1024, band=(0.005, 0.05))
    assert res.n_bands > 3
    assert 0.9 < res.coherence <= 1.0
    assert res.lag_err > 0
    assert res.lag_err < 0.05


def test_band_lag_error_grows_when_the_pair_is_noisy():
    x = _series()
    rng = np.random.default_rng(1)
    clean = _shift(x, 0.5)
    noisy = clean + 3 * rng.standard_normal(x.size) * np.std(clean)
    a = gv.signal.band_lag(x, clean, fs=FS, nperseg=1024, band=(0.005, 0.05))
    b = gv.signal.band_lag(
        x, noisy, fs=FS, nperseg=1024, band=(0.005, 0.05), coh_min=0.0
    )
    assert b.lag_err > a.lag_err
    assert b.coherence < a.coherence


def test_band_lag_returns_nan_when_too_few_bands_qualify():
    x = _series()
    rng = np.random.default_rng(2)
    y = rng.standard_normal(x.size)
    res = gv.signal.band_lag(
        x, y, fs=FS, nperseg=1024, band=(0.005, 0.05), coh_min=0.99
    )
    assert np.isnan(res.lag)
    assert np.isnan(res.lag_err)
    assert res.n_bands < 3


def test_band_lag_rejects_nans():
    x = _series()
    y = _shift(x, 0.5)
    y[100] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        gv.signal.band_lag(x, y, fs=FS, nperseg=1024, band=(0.005, 0.05))


def test_band_lag_rejects_mismatched_lengths():
    x = _series()
    with pytest.raises(ValueError, match="same shape"):
        gv.signal.band_lag(x, x[:-1], fs=FS, nperseg=1024, band=(0.005, 0.05))


def test_band_lag_returns_record_mean_under_a_linear_drift():
    """Welch averages segments, so a ramp from zero reports half its end value."""
    x = _series()
    i = np.arange(x.size, dtype=float)
    end = 4.0
    y = np.interp(i - end * (i / (i.size - 1)), i, x)
    res = gv.signal.band_lag(x, y, fs=FS, nperseg=1024, band=(0.005, 0.05))
    assert res.lag == pytest.approx(end / 2, rel=0.15)
