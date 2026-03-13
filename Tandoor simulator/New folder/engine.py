# engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Protocol
import numpy as np


@dataclass
class SimulationConfig:
    n_servers: int
    n_customers: int
    seed: int = 42


@dataclass
class Customer:
    id: int
    arrival: float


class Discipline(Protocol):
    name: str
    def reset(self) -> None: ...
    def push(self, customer: Customer) -> None: ...
    def pop(self) -> Customer: ...
    def __len__(self) -> int: ...


class ArrivalProcess(Protocol):
    def reset(self) -> None: ...
    def next_arrival_time(self, rng: np.random.Generator, i: int, last_arrival: Optional[float]) -> Optional[float]: ...


class ServiceProcess(Protocol):
    def reset(self) -> None: ...
    def sample_service_time(self, rng: np.random.Generator, customer: Customer, server_id: int) -> float: ...


def simulate(
    cfg: SimulationConfig,
    arrivals: ArrivalProcess,
    service: ServiceProcess,
    discipline: Discipline
) -> Dict[str, Any]:
    """
    Discrete-event simulation of a single FIFO queue feeding c servers.
    Returns a dict with event logs, server timelines, queue length time series, and KPIs.
    """
    rng = np.random.default_rng(int(cfg.seed))
    arrivals.reset()
    service.reset()
    discipline.reset()

    c = int(cfg.n_servers)
    N = int(cfg.n_customers)

    # Server state
    server_busy_until = [0.0 for _ in range(c)]
    server_current_customer = [None for _ in range(c)]  # id or None

    # Logs
    events: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []
    ts: List[Dict[str, Any]] = []

    # Queue/system length tracking (time-average)
    last_t = 0.0
    area_Lq = 0.0
    area_L = 0.0

    # Utilization
    busy_time = [0.0 for _ in range(c)]

    # Next arrival
    last_arrival_time: Optional[float] = None
    next_arrival_time = arrivals.next_arrival_time(rng, 1, last_arrival_time)

    # Future departures: we will compute on the fly by checking earliest busy_until among servers
    completed = 0
    created = 0
    in_system = 0  # queue + in service

    # helper
    def record_time_areas(new_t: float):
        nonlocal last_t, area_Lq, area_L
        dt = max(0.0, new_t - last_t)
        Lq = len(discipline)
        L = Lq + sum(1 for x in server_current_customer if x is not None)
        area_Lq += Lq * dt
        area_L += L * dt
        last_t = new_t
        ts.append({"time": new_t, "queue_length": Lq, "system_length": L})

    # initialize time series at t=0
    record_time_areas(0.0)

    # main loop
    while completed < N:
        # find next departure event (earliest server finish)
        dep_t = None
        dep_server = None
        for s in range(c):
            if server_current_customer[s] is not None:
                t_finish = server_busy_until[s]
                if dep_t is None or t_finish < dep_t:
                    dep_t = t_finish
                    dep_server = s

        # decide which event occurs next: arrival vs departure
        if next_arrival_time is not None and (dep_t is None or next_arrival_time <= dep_t) and created < N:
            # ARRIVAL event
            t = float(next_arrival_time)
            record_time_areas(t)

            created += 1
            cust = Customer(id=created, arrival=t)
            in_system += 1

            # If any server free at time t, start service immediately
            free_server = None
            for s in range(c):
                if server_current_customer[s] is None or server_busy_until[s] <= t:
                    # if server finished earlier but not processed? handle by clearing
                    server_current_customer[s] = None if server_busy_until[s] <= t else server_current_customer[s]
                    if server_current_customer[s] is None:
                        free_server = s
                        break

            if free_server is not None:
                s = free_server
                st = t
                svc = float(service.sample_service_time(rng, cust, s + 1))
                et = st + svc
                server_current_customer[s] = cust.id
                server_busy_until[s] = et
                busy_time[s] += svc

                events.append({
                    "Customer_ID": cust.id,
                    "Arrival": t,
                    "ServiceStart": st,
                    "ServiceEnd": et,
                    "ServiceTime": svc,
                    "WaitTime": st - t,
                    "TurnaroundTime": et - t,
                    "Server_ID": s + 1
                })
                timeline.append({
                    "Server_ID": s + 1,
                    "Customer_ID": cust.id,
                    "BusyStart": st,
                    "BusyDuration": svc
                })
            else:
                # join FIFO queue
                discipline.push(cust)

            last_arrival_time = t
            next_arrival_time = arrivals.next_arrival_time(rng, created + 1, last_arrival_time)

        else:
            # DEPARTURE event
            if dep_t is None or dep_server is None:
                # no departures scheduled but arrivals finished (edge)
                break

            t = float(dep_t)
            record_time_areas(t)

            s = dep_server
            finished_cust_id = server_current_customer[s]
            server_current_customer[s] = None
            completed += 1
            in_system -= 1

            # Start next from queue if any
            if len(discipline) > 0:
                next_cust = discipline.pop()
                in_system += 1  # popped from queue -> in service (system count stays same, but we had decreased at depart; we re-add)
                st = t
                svc = float(service.sample_service_time(rng, next_cust, s + 1))
                et = st + svc
                server_current_customer[s] = next_cust.id
                server_busy_until[s] = et
                busy_time[s] += svc

                events.append({
                    "Customer_ID": next_cust.id,
                    "Arrival": next_cust.arrival,
                    "ServiceStart": st,
                    "ServiceEnd": et,
                    "ServiceTime": svc,
                    "WaitTime": st - next_cust.arrival,
                    "TurnaroundTime": et - next_cust.arrival,
                    "Server_ID": s + 1
                })
                timeline.append({
                    "Server_ID": s + 1,
                    "Customer_ID": next_cust.id,
                    "BusyStart": st,
                    "BusyDuration": svc
                })

    # Final time horizon (last_t)
    T = max(last_t, 1e-9)
    utilization = {f"S{s+1}": (busy_time[s] / T) for s in range(c)}

    # KPIs
    if events:
        avg_wait = float(np.mean([e["WaitTime"] for e in events]))
        avg_service = float(np.mean([e["ServiceTime"] for e in events]))
        avg_turn = float(np.mean([e["TurnaroundTime"] for e in events]))
    else:
        avg_wait = avg_service = avg_turn = 0.0

    kpis = {
        "avg_wait": avg_wait,
        "avg_service": avg_service,
        "avg_turnaround": avg_turn,
        "Lq_time_avg": area_Lq / T,
        "L_time_avg": area_L / T,
        "throughput_per_time": (len(events) / T) if T > 0 else 0.0,
        "utilization": utilization,
        "time_horizon": T,
        "customers_completed": len(events),
    }

    return {"events": events, "timeline": timeline, "timeseries": ts, "kpis": kpis}
