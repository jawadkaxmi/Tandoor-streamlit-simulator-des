from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd

from .models import KendallModel

# -----------------------------
# Internal: queue length series
# -----------------------------
def _queue_timeseries(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build queue length over time using event changes:
      +1 at Arrival
      -1 at ServiceStart  (customer leaves queue when service starts)
    This gives queue length (not system length).
    """
    if events.empty:
        return pd.DataFrame({"time": [], "queue_length": []})

    changes = []
    for _, r in events.iterrows():
        changes.append((float(r["Arrival"]), +1))
        changes.append((float(r["ServiceStart"]), -1))

    changes.sort(key=lambda x: (x[0], -x[1]))  # arrivals processed before starts at same time
    t_list, q_list = [], []
    q = 0
    for t, dq in changes:
        q += dq
        q = max(q, 0)
        t_list.append(t)
        q_list.append(q)

    # compress duplicates times (keep last q at time)
    out = pd.DataFrame({"time": t_list, "queue_length": q_list})
    out = out.groupby("time", as_index=False).last()
    return out


# -----------------------------
# FCFS single shared queue, c servers
# -----------------------------
def _simulate_fcfs_shared_queue(
    arrival_times: np.ndarray,
    service_times: np.ndarray,
    n_servers: int,
) -> Dict[str, Any]:
    n = len(arrival_times)
    server_free = np.zeros(int(n_servers), dtype=float)

    rows = []
    tl = []
    for i in range(n):
        arr = float(arrival_times[i])
        svc = float(service_times[i])

        # Next free server
        sid = int(np.argmin(server_free))  # 0-based
        start = max(arr, server_free[sid])
        end = start + svc
        wait = start - arr

        rows.append({
            "Customer_ID": i + 1,
            "Arrival": arr,
            "ServiceStart": start,
            "ServiceEnd": end,
            "ServiceTime": svc,
            "WaitTime": wait,
            "Server_ID": sid + 1,
        })
        tl.append({
            "Server_ID": sid + 1,
            "Customer_ID": i + 1,
            "BusyStart": start,
            "BusyEnd": end,
            "BusyDuration": svc,
        })
        server_free[sid] = end

    events = pd.DataFrame(rows)
    timeline = pd.DataFrame(tl)
    makespan = float(events["ServiceEnd"].max()) if n else 0.0

    util = {}
    for sid in range(1, int(n_servers) + 1):
        busy = float(timeline.loc[timeline["Server_ID"] == sid, "BusyDuration"].sum())
        util[sid] = (busy / makespan) if makespan > 0 else 0.0

    kpis = {
        "n": int(n),
        "time_horizon": makespan,
        "avg_wait": float(events["WaitTime"].mean()) if n else 0.0,
        "avg_service": float(events["ServiceTime"].mean()) if n else 0.0,
        "avg_system_time": float((events["ServiceEnd"] - events["Arrival"]).mean()) if n else 0.0,
        "utilization": util,
    }

    queue_ts = _queue_timeseries(events)

    return {
        "events": events,
        "timeline": timeline,
        "queue_ts": queue_ts,
        "kpis": kpis,
    }


def _servers_from_model(model: KendallModel, default_c: int) -> int:
    if model.servers == "1":
        return 1
    if model.servers == "2":
        return 2
    return int(default_c)


def simulate_tandem_two_stage(
    arrival_times_stage1: np.ndarray,
    service_times_stage1: np.ndarray,
    service_times_stage2: np.ndarray,
    model_stage1: KendallModel,
    model_stage2: KendallModel,
    default_c_stage2: int = 2,
) -> Dict[str, Any]:
    """
    Two-stage tandem system:
      Stage 1: Token counter
      Stage 2: Tandoor (parallel servers, shared queue)
    Stage 2 arrivals = Stage 1 service end times (departure process).
    """
    n = len(arrival_times_stage1)

    c1 = _servers_from_model(model_stage1, default_c=1)
    c2 = _servers_from_model(model_stage2, default_c=default_c_stage2)

    stage1 = _simulate_fcfs_shared_queue(arrival_times_stage1, service_times_stage1, n_servers=c1)

    # Stage 2 arrival = stage1 departures
    ev1 = stage1["events"].sort_values("Customer_ID").reset_index(drop=True)
    arr2 = ev1["ServiceEnd"].to_numpy(dtype=float)

    stage2 = _simulate_fcfs_shared_queue(arr2, service_times_stage2, n_servers=c2)

    ev2 = stage2["events"].sort_values("Customer_ID").reset_index(drop=True)

    combined = pd.DataFrame({
        "Customer_ID": ev1["Customer_ID"],
        "Arrival_Stage1": ev1["Arrival"],
        "Token_Start": ev1["ServiceStart"],
        "Token_End": ev1["ServiceEnd"],
        "Token_ServiceTime": ev1["ServiceTime"],
        "Wait_Token": ev1["WaitTime"],
        "Token_Server": ev1["Server_ID"],

        "Arrival_Stage2": ev2["Arrival"],
        "Tandoor_Start": ev2["ServiceStart"],
        "Tandoor_End": ev2["ServiceEnd"],
        "Tandoor_ServiceTime": ev2["ServiceTime"],
        "Wait_Tandoor": ev2["WaitTime"],
        "Tandoor_Server": ev2["Server_ID"],

        "Total_System_Time": (ev2["ServiceEnd"] - ev1["Arrival"]),
    })

    overall = {
        "n": int(n),
        "model_stage1": model_stage1.label,
        "model_stage2": model_stage2.label,
        "avg_wait_token": float(combined["Wait_Token"].mean()),
        "avg_wait_tandoor": float(combined["Wait_Tandoor"].mean()),
        "avg_total_system_time": float(combined["Total_System_Time"].mean()),
        "token_utilization": stage1["kpis"]["utilization"],
        "tandoor_utilization": stage2["kpis"]["utilization"],
        "makespan_stage2": float(stage2["kpis"]["time_horizon"]),
        "token_servers": int(c1),
        "tandoor_servers": int(c2),
    }

    return {
        "stage1": stage1,
        "stage2": stage2,
        "combined": combined,
        "overall": overall,
    }
