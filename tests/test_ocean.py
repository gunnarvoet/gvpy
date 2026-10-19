import gsw
import numpy as np

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
