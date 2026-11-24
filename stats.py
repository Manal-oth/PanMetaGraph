"""Statistical tests used in Section 7.7 / Table 8.

- paired two-sided t-test over seeds (Table 8 p-values)
- Student-t confidence interval for the paired mean difference
- BCa bootstrap CI (2000 resamples, Table 5)
- Holm-Bonferroni correction over the family of 12 comparisons in Table 8
"""
import math
import numpy as np

try:
    from scipy import stats as _sp
except Exception:                                  # scipy optional
    _sp = None


def _t_sf(t, df):
    if _sp is not None:
        return float(_sp.t.sf(abs(t), df))
    x = df / (df + t * t)                          # regularised incomplete beta
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m, num = i // 2, 0.0
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def _t_ppf(p, df):
    if _sp is not None:
        return float(_sp.t.ppf(p, df))
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1.0 - _t_sf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired_t_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return float(d.mean()), 0.0, 0.0
    t = d.mean() / (sd / math.sqrt(n))
    return float(d.mean()), float(t), float(2 * _t_sf(t, n - 1))


def paired_t_ci(a, b, alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    hw = _t_ppf(1 - alpha / 2, n - 1) * d.std(ddof=1) / math.sqrt(n)
    return float(d.mean() - hw), float(d.mean() + hw)


def ci_from_sds(m1, s1, m2, s2, n=10, alpha=0.05):
    """CI for the mean difference when only per-arm means and SDs are on record.

    This is the construction documented in the Table 8 caption: the arms are
    treated as independent, so se = sqrt(s1^2 + s2^2)/sqrt(n). Use
    `paired_t_ci` whenever the per-seed values themselves are available, since
    the paired interval is tighter.
    """
    se = math.sqrt(s1 * s1 + s2 * s2) / math.sqrt(n)
    hw = _t_ppf(1 - alpha / 2, n - 1) * se
    d = m1 - m2
    return float(d - hw), float(d + hw)


def bca_bootstrap_ci(a, b, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    theta = d.mean()
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    prop = np.mean(boot < theta)
    prop = min(max(prop, 1.0 / (2 * n_boot)), 1 - 1.0 / (2 * n_boot))
    z0 = _norm_ppf(prop)
    jack = np.array([np.delete(d, i).mean() for i in range(n)])
    mj = jack.mean()
    num = ((mj - jack) ** 3).sum()
    den = 6.0 * (((mj - jack) ** 2).sum() ** 1.5)
    acc = num / den if den else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = _norm_ppf(q)
        adj = z0 + (z0 + z) / max(1 - acc * (z0 + z), 1e-12)
        out.append(float(np.quantile(boot, _norm_cdf(adj))))
    return out[0], out[1]


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def holm_bonferroni(pvals, alpha=0.05):
    """Returns (adjusted p-values, reject flags) for a family of size len(pvals).
    Table 8 reports a family of 12 comparisons (4 cohorts x 3 metrics)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj.tolist(), (adj <= alpha).tolist()
