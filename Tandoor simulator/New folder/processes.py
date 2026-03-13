# processes.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, List
import numpy as np
from engine import ArrivalProcess, ServiceProcess, Customer


@dataclass
class PoissonArrival(ArrivalProcess):
    """Poisson arrivals => exponential inter-arrival with rate λ."""
    lam: float  # arrivals per minute

    def reset(self) -> None:
        pass

    def next_arrival_time(self, rng: np.random.Generator, i: int, last_arrival: Optional[float]) -> Optional[float]:
        if self.lam <= 0:
            return None
        last = 0.0 if last_arrival is None else float(last_arrival)
        ia = float(rng.exponential(1.0 / self.lam))
        return last + ia


@dataclass
class EmpiricalArrivalTimes(ArrivalProcess):
    """Deterministic replay using exact arrival times (minutes)."""
    arrival_times: Sequence[float]

    def reset(self) -> None:
        self._times = [float(t) for t in self.arrival_times if t is not None and np.isfinite(float(t))]
        self._times.sort()
        self._idx = 0

    def next_arrival_time(self, rng: np.random.Generator, i: int, last_arrival: Optional[float]) -> Optional[float]:
        if self._idx >= len(self._times):
            return None
        t = float(self._times[self._idx])
        self._idx += 1
        return t


@dataclass
class EmpiricalInterarrival(ArrivalProcess):
    """G-arrivals: sample inter-arrival times from data."""
    interarrivals: Sequence[float]

    def reset(self) -> None:
        vals = []
        for x in self.interarrivals:
            if x is None:
                continue
            v = float(x)
            if np.isfinite(v) and v > 0:
                vals.append(v)
        if not vals:
            raise ValueError("EmpiricalInterarrival: no valid inter-arrival values.")
        self._vals: List[float] = vals

    def next_arrival_time(self, rng: np.random.Generator, i: int, last_arrival: Optional[float]) -> Optional[float]:
        last = 0.0 if last_arrival is None else float(last_arrival)
        ia = float(rng.choice(self._vals))
        return last + ia


@dataclass
class ExponentialService(ServiceProcess):
    """Exponential service with rate μ."""
    mu: float  # services per minute per server

    def reset(self) -> None:
        pass

    def sample_service_time(self, rng: np.random.Generator, customer: Customer, server_id: int) -> float:
        if self.mu <= 0:
            return 0.0
        return float(rng.exponential(1.0 / self.mu))


@dataclass
class EmpiricalServiceTimes(ServiceProcess):
    """G-service: sample service durations from data."""
    service_times: Sequence[float]

    def reset(self) -> None:
        vals = []
        for x in self.service_times:
            if x is None:
                continue
            v = float(x)
            if np.isfinite(v) and v > 0:
                vals.append(v)
        if not vals:
            raise ValueError("EmpiricalServiceTimes: no valid service time values.")
        self._vals: List[float] = vals

    def sample_service_time(self, rng: np.random.Generator, customer: Customer, server_id: int) -> float:
        return float(rng.choice(self._vals))
