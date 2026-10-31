from pathlib import Path

import gsw
import numpy as np
import xarray as xr

import gvpy


def test_nsqfcn_matches_gsw_nsquared_on_smooth_profile():
    p = np.arange(0, 1000.5, 0.5)
    t = 20 - 16 * p / 1000
    s = 34 + p / 1000
    lon, lat = -125.0, 45.0
    n2, pout = gvpy.ocean.nsqfcn(s, t, p, p0=0, dp=10, lon=lon, lat=lat)
    SA = gsw.SA_from_SP(s, p, lon, lat)
    CT = gsw.CT_from_t(SA, t, p)
    n2_gsw, p_mid = gsw.Nsquared(SA, CT, p, lat)
    ref = np.interp(pout, p_mid, n2_gsw)
    assert (n2 > 0).all()
    # dp is in dbar where gsw differences in Pa, about 1 % apart at depth.
    np.testing.assert_allclose(n2, ref, rtol=0.03)


def test_nsqfcn_coarse_profile_returns_finite_n2():
    # 6 dbar spacing with dp=10 used to build np.hanning(2), all zeros
    p = np.arange(0, 1000.5, 6.0)
    t = 20 - 16 * p / 1000
    s = 34 + p / 1000
    n2, pout = gvpy.ocean.nsqfcn(s, t, p, p0=0, dp=10, lon=-125.0, lat=45.0)
    assert np.isfinite(n2).all()


def test_nsqfcn_all_nan_returns_nan():
    """Regression test: the no-good-data branch fell through and raised IndexError."""
    p = np.arange(0, 100.0)
    nan = np.full_like(p, np.nan)
    n2, pout = gvpy.ocean.nsqfcn(nan, nan, p, p0=0, dp=10, lon=0, lat=45)
    assert np.isnan(n2) and np.isnan(pout)


def test_tzfcn_all_nan_returns_nan_of_input_shape():
    """Regression test: the no-good-data branch fell through and raised."""
    z = np.arange(0, 100.0)
    tz = gvpy.ocean.tzfcn(np.full_like(z, np.nan), z, z0=0, dz=10)
    assert tz.shape == z.shape
    assert np.isnan(tz).all()


def test_tzfcn_non_monotonic_returns_nan(capsys):
    """Regression test: the non-monotonic branch raised UnboundLocalError."""
    z = np.concatenate([np.arange(0, 40.0), np.arange(20, 80.0)])
    tz = gvpy.ocean.tzfcn(10 - z / 100, z, z0=0, dz=5)
    assert "not monotonic" in capsys.readouterr().out
    assert tz.shape == z.shape
    assert np.isnan(tz).all()


def test_wind_stress_scalar_matches_array():
    """Regression test: scalar wind raised ValueError on np.where of a 0-d array."""
    u = np.array([0.5, 2.0, 5.0, 12.0])
    v = np.zeros_like(u)
    tx, ty = gvpy.ocean.wind_stress(u, v)
    for ui, txi in zip(u, tx):
        txs, tys = gvpy.ocean.wind_stress(ui, 0.0)
        assert np.ndim(txs) == 0
        assert txs == txi
        assert tys == 0


def test_smith_sandwell_all_returns_whole_dataset(tmp_path, monkeypatch):
    """Regression test: the default lon='all' was wrapped into a list and never matched."""
    d = tmp_path / "Data/bathymetry/smith_sandwell"
    d.mkdir(parents=True)
    topo = xr.DataArray(
        -np.arange(12.0).reshape(3, 4),
        coords=dict(lat=[-1.0, 0.0, 1.0], lon=[0.0, 90.0, 180.0, 270.0]),
        dims=("lat", "lon"),
        name="z",
    )
    topo.to_netcdf(d / "SRTM15_V2.6.nc", engine="netcdf4")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    b = gvpy.ocean.smith_sandwell()
    assert b.shape == (3, 4)
    np.testing.assert_array_equal(b.lon, [-180.0, -90.0, 0.0, 90.0])


def test_eps_overturn2_runs():
    """Regression test: np.alen was removed from numpy."""
    p = np.arange(0, 500.0)
    t = 20 - 10 * p / 500
    t[100:120] = t[100:120][::-1]
    s = np.full_like(p, 35.0)
    z = -gsw.z_from_p(p, 45)
    out = gvpy.ocean.eps_overturn2(p, z, t, s, 0, 45)
    assert len(out["Lt"]) == len(p)


def test_bathy_section_meridional_segment():
    """Regression test: np.int was removed from numpy."""
    bathy = xr.DataArray(
        -np.ones((11, 11)),
        coords=dict(lat=np.linspace(0, 1, 11), lon=np.linspace(0, 1, 11)),
        dims=("lat", "lon"),
    )
    gvpy.ocean.bathy_section(bathy, np.array([0.5, 0.5]), np.array([0.1, 0.9]), res=0.1)


def test_nsqfcn_sort_uses_pressure_for_absolute_salinity(monkeypatch):
    """Regression test: sort=True passed temperature as pressure to SA_from_SP."""
    p = np.arange(0, 1000.5, 0.5)
    t = 20 - 16 * p / 1000
    s = 34 + p / 1000
    calls = []
    SA_from_SP = gsw.SA_from_SP

    def spy(SP, pressure, lon, lat):
        calls.append(np.asarray(pressure))
        return SA_from_SP(SP, pressure, lon, lat)

    monkeypatch.setattr(gsw, "SA_from_SP", spy)
    n2, pout = gvpy.ocean.nsqfcn(s, t, p, p0=0, dp=10, lon=-125, lat=45, sort=True)
    # the sorting call gets the full filtered pressure profile
    assert any(c.size == p.size and np.allclose(c, p, atol=1) for c in calls)
    assert not any(c.size == p.size and np.allclose(c, t, atol=1) for c in calls)
    assert (n2 > 0).all()
