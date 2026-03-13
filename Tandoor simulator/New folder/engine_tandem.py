# engine_tandem.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

from engine import SimulationConfig, simulate, Customer
from processes import EmpiricalArrivalTimes


@dataclass
class TandemConfig:
    n_customers: int
    seed: int = 42
    # Stage capacities
    token_servers: int = 1
    tandoor_servers: int = 2


def simulate_tandem(
    cfg: TandemConfig,
    arrivals_stage1,
    service_stage1,
    service_stage2,
    discipline_stage1,
    discipline_stage2,
) -> Dict[str, Any]:
    """
    Two-stage tandem:
      Stage 1: Token counter (c=1 default)
      Stage 2: Tandoor (c=2 default, shared FIFO queue)

    Stage 2 arrivals are the Stage 1 ServiceEnd times.
    """

    # ---- Stage 1 ----
    cfg1 = SimulationConfig(n_servers=int(cfg.token_servers),
                            n_customers=int(cfg.n_customers),
                            seed=int(cfg.seed))

    sim1 = simulate(cfg1, arrivals_stage1, service_stage1, discipline_stage1)
    ev1 = sim1.get("events", [])

    if len(ev1) < cfg.n_customers:
        raise ValueError(f"Stage 1 completed {len(ev1)} customers, expected {cfg.n_customers}.")

    # Sort by Customer_ID to align customers (1..N)
    ev1_sorted = sorted(ev1, key=lambda r: int(r["Customer_ID"]))
    stage2_arrivals = np.array([float(r["ServiceEnd"]) for r in ev1_sorted], dtype=float)

    # ---- Stage 2 arrivals = deterministic replay of stage2_arrivals ----
    arrivals_stage2 = EmpiricalArrivalTimes(arrival_times=stage2_arrivals)

    cfg2 = SimulationConfig(n_servers=int(cfg.tandoor_servers),
                            n_customers=int(cfg.n_customers),
                            seed=int(cfg.seed) + 1)  # different stream

    sim2 = simulate(cfg2, arrivals_stage2, service_stage2, discipline_stage2)
    ev2 = sim2.get("events", [])

    if len(ev2) < cfg.n_customers:
        raise ValueError(f"Stage 2 completed {len(ev2)} customers, expected {cfg.n_customers}.")

    ev2_sorted = sorted(ev2, key=lambda r: int(r["Customer_ID"]))

    # ---- Combine customer-level view ----
    combined = []
    for i in range(cfg.n_customers):
        r1 = ev1_sorted[i]
        r2 = ev2_sorted[i]

        combined.append({
            "Customer_ID": int(r1["Customer_ID"]),

            # Stage 1 (Token)
            "Arrival_Stage1": float(r1["Arrival"]),
            "Token_ServiceStart": float(r1["ServiceStart"]),
            "Token_ServiceEnd": float(r1["ServiceEnd"]),
            "Token_ServiceTime": float(r1["ServiceTime"]),
            "Wait_Token": float(r1["WaitTime"]),
            "Token_Server": int(r1["Server_ID"]),

            # Stage 2 (Tandoor)
            "Arrival_Stage2": float(r2["Arrival"]),  # equals Token_ServiceEnd
            "Tandoor_ServiceStart": float(r2["ServiceStart"]),
            "Tandoor_ServiceEnd": float(r2["ServiceEnd"]),
            "Tandoor_ServiceTime": float(r2["ServiceTime"]),
            "Wait_Tandoor": float(r2["WaitTime"]),
            "Tandoor_Server": int(r2["Server_ID"]),  # can be hidden in UI/export if you want

            # Overall
            "Total_System_Time": float(r2["ServiceEnd"] - r1["Arrival"]),
        })

    # KPI summary (stage-wise + overall)
    k1 = sim1.get("kpis", {}) or {}
    k2 = sim2.get("kpis", {}) or {}

    overall = {
        "n": int(cfg.n_customers),
        "token_avg_wait": float(k1.get("avg_wait", 0.0)),
        "tandoor_avg_wait": float(k2.get("avg_wait", 0.0)),
        "total_avg_system_time": float(np.mean([x["Total_System_Time"] for x in combined])),
        "token_utilization": k1.get("utilization", {}),
        "tandoor_utilization": k2.get("utilization", {}),
        "token_time_horizon": float(k1.get("time_horizon", 0.0)),
        "tandoor_time_horizon": float(k2.get("time_horizon", 0.0)),
    }

    return {
        "stage1": sim1,
        "stage2": sim2,
        "combined": combined,
        "overall_kpis": overall,
    }
