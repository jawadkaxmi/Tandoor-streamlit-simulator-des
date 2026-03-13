# core/models.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class KendallModel:
    """
    Represents Kendall notation model: A/B/c
      arrival: 'M' or 'G'
      service: 'M' or 'G'
      servers: '1' or '2' or 'c'
    """
    arrival: str
    service: str
    servers: str

    @property
    def label(self) -> str:
        return f"{self.arrival}/{self.service}/{self.servers}"

    @property
    def compact(self) -> str:
        # MM1 / MG2 / GGC
        s = self.servers.upper() if self.servers.lower() == "c" else self.servers
        return f"{self.arrival}{self.service}{s}"


# ✅ UI expects .label on each element
ALL_9_MODELS = [
    KendallModel("M", "M", "1"),
    KendallModel("M", "M", "2"),
    KendallModel("M", "M", "c"),
    KendallModel("M", "G", "1"),
    KendallModel("M", "G", "2"),
    KendallModel("M", "G", "c"),
    KendallModel("G", "G", "1"),
    KendallModel("G", "G", "2"),
    KendallModel("G", "G", "c"),
]


def parse_label(label: str) -> KendallModel:
    """
    Accepts:
      - "M/M/1"
      - "MM1", "MG2", "GGC"
    Returns KendallModel object.
    """
    s = str(label).strip().upper().replace(" ", "")
    if "/" in s:
        a, b, c = s.split("/")
        c = "c" if c == "C" else c
        return KendallModel(a, b, c)

    if len(s) != 3:
        raise ValueError(f"Invalid model label: {label}")

    a = s[0]  # M or G
    b = s[1]  # M or G
    c = s[2]  # 1 / 2 / C
    c = "c" if c == "C" else c
    return KendallModel(a, b, c)


def build_arrival_times(
    model: KendallModel,
    n: int,
    rng: np.random.Generator,
    mean_interarrival: float | None = None,
    empirical_arrival_times: np.ndarray | None = None
) -> np.ndarray:
    """
    Returns arrival TIMES (minutes, starting at 0).
    For M: exponential inter-arrivals with mean_interarrival
    For G: resample empirical inter-arrivals derived from empirical_arrival_times
    """
    n = int(n)
    if n <= 0:
        return np.array([], dtype=float)

    if model.arrival == "M":
        if mean_interarrival is None or mean_interarrival <= 0:
            mean_interarrival = 1.0
        ia = rng.exponential(scale=float(mean_interarrival), size=n)
        return np.cumsum(ia).astype(float)

    # G arrivals
    if empirical_arrival_times is None:
        # fallback
        if mean_interarrival is None or mean_interarrival <= 0:
            mean_interarrival = 1.0
        ia = rng.exponential(scale=float(mean_interarrival), size=n)
        return np.cumsum(ia).astype(float)

    arr = np.asarray(empirical_arrival_times, dtype=float)
    arr = np.sort(arr[np.isfinite(arr)])
    ia_pool = np.diff(arr)
    ia_pool = ia_pool[np.isfinite(ia_pool) & (ia_pool > 0)]

    if len(ia_pool) == 0:
        if mean_interarrival is None or mean_interarrival <= 0:
            mean_interarrival = 1.0
        ia = rng.exponential(scale=float(mean_interarrival), size=n)
        return np.cumsum(ia).astype(float)

    ia = rng.choice(ia_pool, size=n, replace=True).astype(float)
    return np.cumsum(ia).astype(float)


def build_service_times(
    model: KendallModel,
    n: int,
    rng: np.random.Generator,
    mean_service: float | None = None,
    empirical_service_times: np.ndarray | None = None
) -> np.ndarray:
    """
    Returns service TIMES (minutes).
    For M: exponential service with mean_service
    For G: resample from empirical_service_times
    """
    n = int(n)
    if n <= 0:
        return np.array([], dtype=float)

    if model.service == "M":
        if mean_service is None or mean_service <= 0:
            mean_service = 1.0
        return rng.exponential(scale=float(mean_service), size=n).astype(float)

    # G services
    if empirical_service_times is None:
        if mean_service is None or mean_service <= 0:
            mean_service = 1.0
        return rng.exponential(scale=float(mean_service), size=n).astype(float)

    s = np.asarray(empirical_service_times, dtype=float)
    s = s[np.isfinite(s) & (s >= 0)]

    if len(s) == 0:
        if mean_service is None or mean_service <= 0:
            mean_service = 1.0
        return rng.exponential(scale=float(mean_service), size=n).astype(float)

    return rng.choice(s, size=n, replace=True).astype(float)
