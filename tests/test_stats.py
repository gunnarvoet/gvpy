"""Tests for gvpy.stats."""

import numpy as np
from scipy.stats import norm

import gvpy as gv


def test_median_ratio_ci():
    rng = np.random.default_rng(0)
    b = np.exp(rng.normal(0, 0.5, 400))
    a = np.r_[3 * b, np.nan]
    r, lo, hi = gv.stats.median_ratio_ci(a, b, rng=rng)
    np.testing.assert_allclose(r, 3.0)
    assert lo < 3.0 < hi
    # standard error of the median of n normal values is 1.2533 sigma / sqrt(n)
    se = 1.2533 * 0.5 / np.sqrt(400) * np.sqrt(2)
    np.testing.assert_allclose(np.log(hi / lo), 2 * 1.96 * se, rtol=0.25)


def test_min_detectable_ratio_normal():
    # Lognormal samples with sigma = 0.8 and 35 values each. A t-test detects a
    # shift of (z_0.975 + z_0.8) sigma sqrt(2 / n) with 80% power. The
    # Mann-Whitney test has an asymptotic relative efficiency of 3 / pi.
    n, sigma = 35, 0.8
    x = np.exp(sigma * norm.ppf((np.arange(n) + 0.5) / n))
    expected = (norm.ppf(0.975) + norm.ppf(0.8)) * sigma * np.sqrt(2 / n)
    expected = np.exp(expected / np.sqrt(3 / np.pi))
    r = gv.stats.min_detectable_ratio(x, 2 * x, rng=np.random.default_rng(0))
    np.testing.assert_allclose(r, expected, rtol=0.08)


def test_min_detectable_ratio_limits():
    rng = np.random.default_rng(0)
    x = np.exp(rng.normal(0, 0.8, 35))
    # a narrower distribution and a larger sample both lower the limit
    r0 = gv.stats.min_detectable_ratio(x, x, rng=np.random.default_rng(1))
    r1 = gv.stats.min_detectable_ratio(x**0.5, x**0.5, rng=np.random.default_rng(1))
    r2 = gv.stats.min_detectable_ratio(
        np.tile(x, 4), np.tile(x, 4), rng=np.random.default_rng(1)
    )
    assert r1 < r0 and r2 < r0
    # not reached within the trial ratios
    assert np.isnan(gv.stats.min_detectable_ratio(x, x, ratios=[1.01, 1.02]))


def test_coherent_fraction():
    rng = np.random.default_rng(0)
    lev = np.arange(40)[:, None]
    # the same profile in all casts
    same = np.tile(np.sin(lev / 5.0), (1, 50))
    f, lo, hi, chance, n = gv.stats.coherent_fraction(same, rng=rng)
    np.testing.assert_allclose([f, lo, hi], 1.0)
    assert n == 50 and chance == 1 / 50
    # random phase: close to 1 / n
    phase = rng.uniform(0, 2 * np.pi, 400)
    f, lo, hi, chance, n = gv.stats.coherent_fraction(
        np.sin(lev / 5.0 + phase), rng=rng
    )
    assert f < 4 * chance
    # equal variance in a common profile and in random-phase waves
    mixed = same[:, :1] + np.sin(lev / 5.0 + phase)
    f, lo, hi, chance, n = gv.stats.coherent_fraction(mixed, rng=rng)
    np.testing.assert_allclose(f, 0.5, atol=0.06)
    assert lo < 0.5 < hi
    # a cast with a gap is ignored
    same[3, 0] = np.nan
    assert gv.stats.coherent_fraction(same, rng=rng)[4] == 49
