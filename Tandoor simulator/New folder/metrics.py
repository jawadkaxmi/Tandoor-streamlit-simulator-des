# metrics.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def to_dataframes(sim: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_df = pd.DataFrame(sim.get("events", []))
    timeline_df = pd.DataFrame(sim.get("timeline", []))
    ts_df = pd.DataFrame(sim.get("timeseries", []))
    k = sim.get("kpis", {}) or {}
    kpi_df = pd.DataFrame([k]) if k else pd.DataFrame()
    return event_df, timeline_df, kpi_df, ts_df


def chi_square_exponential_gof(
    x: np.ndarray,
    bins: int = 10,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Chi-square GOF for Exponential distribution with MLE rate λ = 1/mean(x).
    Uses equal-probability bins via fitted CDF quantiles to satisfy expected counts.

    H0: data ~ Exponential(λ_hat)
    H1: not Exponential
    """
    if not SCIPY_OK:
        return {"ok": False, "error": "scipy not installed. Install: py -m pip install scipy"}

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]

    n = len(x)
    if n < 30:
        return {"ok": False, "error": f"Need at least ~30 samples for a stable chi-square test. Got n={n}."}

    lam_hat = 1.0 / float(np.mean(x))

    # Choose bins so expected count in each bin = n/bins
    # Quantiles for exponential: Q(p) = -ln(1-p)/λ
    ps = np.linspace(0, 1, bins + 1)
    # avoid 0 and 1 exactly
    ps[0] = 0.0
    ps[-1] = 1.0

    edges = []
    for p in ps:
        if p <= 0:
            edges.append(0.0)
        elif p >= 1:
            edges.append(np.inf)
        else:
            edges.append(float(-np.log(1 - p) / lam_hat))

    # Observed counts
    obs = np.zeros(bins, dtype=int)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if np.isinf(hi):
            obs[i] = int(np.sum(x >= lo))
        else:
            obs[i] = int(np.sum((x >= lo) & (x < hi)))

    exp = np.full(bins, n / bins, dtype=float)

    # Chi-square statistic
    chi_stat = float(np.sum((obs - exp) ** 2 / exp))

    # df = bins - 1 - p (p=1 parameter estimated)
    df = max(1, bins - 2)
    p_value = float(chi2.sf(chi_stat, df))

    decision = "Fail to reject H0 (Exponential acceptable)" if p_value > alpha else "Reject H0 (Not Exponential)"

    table = []
    for i in range(bins):
        table.append({
            "Bin": i + 1,
            "Lower": edges[i],
            "Upper": edges[i + 1],
            "Observed": int(obs[i]),
            "Expected": float(exp[i]),
        })

    return {
        "ok": True,
        "n": n,
        "lambda_hat": lam_hat,
        "bins": bins,
        "df": df,
        "chi_square": chi_stat,
        "p_value": p_value,
        "alpha": alpha,
        "decision": decision,
        "table": pd.DataFrame(table),
    }
