# core/stats.py

import numpy as np
from scipy.stats import chi2

def chi_square_exponential_gof(samples, bins=10, alpha=0.05):
    """
    Chi-square goodness-of-fit test for Exponential distribution
    Mean is estimated from data → df = k - 2
    """

    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    n = len(x)

    if n < 20:
        return {
            "chi2": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "decision": "Insufficient data"
        }

    mean = np.mean(x)
    if mean <= 0:
        return {
            "chi2": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "decision": "Invalid mean"
        }

    # Equal probability bins (BEST PRACTICE)
    q = np.linspace(0, 1, bins + 1)
    edges = -mean * np.log(1 - q)
    edges[0] = 0
    edges[-1] = max(edges[-1], x.max() * 1.01)

    observed, _ = np.histogram(x, bins=edges)
    expected = np.full(bins, n / bins)

    # Merge bins if expected < 5
    obs_m, exp_m = [], []
    o_acc, e_acc = 0, 0

    for o, e in zip(observed, expected):
        o_acc += o
        e_acc += e
        if e_acc >= 5:
            obs_m.append(o_acc)
            exp_m.append(e_acc)
            o_acc, e_acc = 0, 0

    if e_acc > 0:
        obs_m[-1] += o_acc
        exp_m[-1] += e_acc

    obs_m = np.array(obs_m)
    exp_m = np.array(exp_m)

    chi_stat = np.sum((obs_m - exp_m) ** 2 / exp_m)

    # df = bins - 1 - parameters estimated (1)
    df = max(1, len(obs_m) - 2)

    p_value = 1 - chi2.cdf(chi_stat, df)

    decision = "Fail to reject H0" if p_value > alpha else "Reject H0"

    return {
        "chi2": float(chi_stat),
        "df": int(df),
        "p_value": float(p_value),
        "mean": float(mean),
        "decision": decision
    }
