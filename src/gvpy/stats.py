"""Statistics for comparisons between groups of samples: ratios of medians with
bootstrap ranges, detection limits of two-sample tests, and coherence of
signed profiles.
"""

import numpy as np
from scipy.stats import mannwhitneyu


def _finite(x):
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def median_ratio_ci(a, b, nboot=2000, ci=95.0, rng=None):
    """Ratio of the medians of two samples with a bootstrap range.

    Parameters
    ----------
    a, b : array-like
        Positive samples. Non-finite values are ignored.
    nboot : int, optional
        Number of bootstrap resamples, drawn within each sample. Defaults to
        2000.
    ci : float, optional
        Width of the percentile range [%]. Defaults to 95.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    ratio, lo, hi : float
        median(a) / median(b) and the lower and upper end of the bootstrap
        percentile range.
    """
    rng = np.random.default_rng() if rng is None else rng
    a, b = np.log(_finite(a)), np.log(_finite(b))
    ma = np.median(rng.choice(a, (nboot, a.size)), axis=1)
    mb = np.median(rng.choice(b, (nboot, b.size)), axis=1)
    lo, hi = np.percentile(ma - mb, [50 - ci / 2, 50 + ci / 2])
    return tuple(float(np.exp(x)) for x in (np.median(a) - np.median(b), lo, hi))


def min_detectable_ratio(a, b, power=0.8, alpha=0.05, ratios=None, nsim=2000, rng=None):
    """Smallest multiplicative enhancement of `a` over `b` that a two-sided
    Mann-Whitney test would detect with the given power.

    The two samples are pooled in log space after removing their medians.
    Synthetic samples of the sizes of `a` and `b` are drawn with replacement
    from the pooled sample, the first one is multiplied by a trial ratio, and
    the fraction of draws with p < `alpha` is the power at that ratio. The
    same draws are used for all trial ratios.

    Parameters
    ----------
    a, b : array-like
        Positive samples. Non-finite values are ignored.
    power : float, optional
        Required detection probability. Defaults to 0.8.
    alpha : float, optional
        Significance level of the test. Defaults to 0.05.
    ratios : array-like, optional
        Increasing trial ratios. Defaults to 80 values from 1.02 to 10,
        logarithmically spaced.
    nsim : int, optional
        Number of synthetic sample pairs. Defaults to 2000.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    float
        Ratio at which the power reaches `power`, interpolated in log ratio.
        NaN if the largest trial ratio does not reach it.
    """
    rng = np.random.default_rng() if rng is None else rng
    ratios = np.geomspace(1.02, 10, 80) if ratios is None else np.asarray(ratios)
    a, b = np.log(_finite(a)), np.log(_finite(b))
    pool = np.concatenate([a - np.median(a), b - np.median(b)])
    x = rng.choice(pool, (nsim, a.size))
    y = rng.choice(pool, (nsim, b.size))
    pw = np.array(
        [np.mean(mannwhitneyu(x + np.log(r), y, axis=1).pvalue < alpha) for r in ratios]
    )
    if pw[-1] < power:
        return np.nan
    i = int(np.argmax(pw >= power))
    if i == 0:
        return float(ratios[0])
    lr = np.interp(power, pw[i - 1 : i + 1], np.log(ratios[i - 1 : i + 1]))
    return float(np.exp(lr))


def coherent_fraction(x, nboot=2000, ci=95.0, rng=None):
    """Fraction of the variance of signed profiles that is held by their mean
    over casts.

    Parameters
    ----------
    x : array-like
        Signed quantity with dimensions (level, cast). Casts with a NaN at any
        level are ignored.
    nboot : int, optional
        Number of bootstrap resamples over casts. Defaults to 2000.
    ci : float, optional
        Width of the percentile range [%]. Defaults to 95.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    fraction, lo, hi, chance, n : float, float, float, float, int
        Sum over levels of the squared cast mean divided by the sum over
        levels of the mean square, its bootstrap range, the value 1 / n
        expected for n casts with independent sign, and n.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, dtype=float)
    x = x[:, np.isfinite(x).all(axis=0)]
    n = x.shape[1]
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, n

    def frac(y):
        return float((y.mean(axis=1) ** 2).sum() / (y**2).mean(axis=1).sum())

    boot = [frac(x[:, rng.integers(0, n, n)]) for _ in range(nboot)]
    lo, hi = np.percentile(boot, [50 - ci / 2, 50 + ci / 2])
    return frac(x), float(lo), float(hi), 1 / n, n
