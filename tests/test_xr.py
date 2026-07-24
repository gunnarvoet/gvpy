import numpy as np
import pytest
import xarray as xr

import gvpy as gv  # noqa: F401  (registers the .gv accessor)


def time_series(sampling_period_s, n=100, start="2025-01-01"):
    """Regularly sampled time series with the given sampling period."""
    time = np.datetime64(start) + (
        np.arange(n) * np.timedelta64(round(sampling_period_s * 1e9), "ns")
    )
    return xr.DataArray(np.arange(n, dtype=float), coords=dict(time=time))


@pytest.mark.parametrize(
    "sampling_period_s",
    [0.0625, 0.5, 1.0, 2.0, 60.0, 600.0],
)
def test_sampling_period_recovers_sampling_period(sampling_period_s):
    """The sampling period is recovered for sub-second and multi-second rates."""
    da = time_series(sampling_period_s)
    assert da.gv.sampling_period == pytest.approx(sampling_period_s)


def test_sampling_period_returns_float():
    """Sub-second sampling must not be truncated to an integer number of seconds."""
    da = time_series(0.5)
    assert isinstance(da.gv.sampling_period, float)


def test_sampling_period_rounds_away_jitter():
    """Timestamp jitter well below the sampling period is rounded away."""
    da = time_series(0.5)
    rng = np.random.default_rng(42)
    jitter = rng.integers(-2000, 2000, size=da.time.size) * np.timedelta64(1, "ns")
    da = da.assign_coords(time=da.time + jitter)
    assert da.gv.sampling_period == 0.5


def test_sampling_period_without_time_dimension_is_none():
    """Without a time dimension there is no sampling period to report."""
    da = xr.DataArray(np.arange(10.0), coords=dict(depth=np.arange(10.0)))
    assert da.gv.sampling_period is None


@pytest.mark.parametrize("sampling_period_s", [0.5, 2.0])
def test_ts_lp_runs_for_sub_second_sampling(sampling_period_s):
    """Filtering derives the sampling frequency from the sampling period."""
    da = time_series(sampling_period_s, n=500)
    out = da.gv.ts_lp(cutoff_period=20 * sampling_period_s)
    assert np.isfinite(out).all()
