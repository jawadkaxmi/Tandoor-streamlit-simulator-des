import simpy
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


# -----------------------------
# Helpers
# -----------------------------
def clock_from_start(start_clock_hms: str, minutes_from_start: float) -> str:
    """Convert minutes-from-start into HH:MM:SS using the dataset start clock time."""
    # start_clock_hms = "19:00:00"
    h, m, s = map(int, start_clock_hms.split(":"))
    start_seconds = h * 3600 + m * 60 + s
    t = start_seconds + int(round(minutes_from_start * 60))
    t = t % (24 * 3600)
    hh = t // 3600
    mm = (t % 3600) // 60
    ss = t % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def ensure_arrival_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      - Arrival_min_from_19 / Arrival_min_from_13 / Arrival_min_from_* (already minutes)
      - OR Arrival_24h only (then converts to minutes-from-first-arrival)
    """
    # If any arrival-min column exists, use it.
    possible = [c for c in df.columns if c.lower().startswith("arrival_min_from_")]
    if possible:
        col = possible[0]
        df = df.copy()
        df["Arrival_min"] = pd.to_numeric(df[col], errors="coerce")
        return df

    # Else, build from Arrival_24h
    if "Arrival_24h" not in df.columns:
        raise ValueError("Dataset must contain Arrival_24h or Arrival_min_from_* column.")

    df = df.copy()
    # Convert HH:MM:SS to seconds-from-midnight
    def hms_to_sec(x: str) -> int:
        h, m, s = map(int, x.split(":"))
        return h * 3600 + m * 60 + s

    secs = df["Arrival_24h"].astype(str).map(hms_to_sec).to_numpy()
    # Assume same-day increasing; if wrap occurs, user should provide minutes column
    base = secs[0]
    mins = (secs - base) / 60.0
    df["Arrival_min"] = mins
    return df


# -----------------------------
# Simulation entities
# -----------------------------
@dataclass
class CustomerLog:
    token_no: int
    arrival_min: float
    interarrival_min: float
    token_service_min: float
    tandoor_service_min: float

    token_start: Optional[float] = None
    token_end: Optional[float] = None
    wait_token: Optional[float] = None

    stage2_arrival: Optional[float] = None
    tandoor_start: Optional[float] = None
    tandoor_end: Optional[float] = None
    wait_tandoor: Optional[float] = None

    assigned_tandoor: Optional[int] = None


class TwoStageTandoorDES:
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.token_counter = simpy.Resource(env, capacity=1)

        # Two tandoors (we’ll select whichever becomes available first)
        self.t1 = simpy.Resource(env, capacity=1)
        self.t2 = simpy.Resource(env, capacity=1)

        # For utilization
        self.busy_token = 0.0
        self.busy_t1 = 0.0
        self.busy_t2 = 0.0

    def _serve(self, service_time: float):
        yield self.env.timeout(service_time)

    def customer_process(self, row: CustomerLog):
        # Arrival
        yield self.env.timeout(max(0.0, row.arrival_min - self.env.now))

        # ---- Stage 1: Token counter ----
        with self.token_counter.request() as req:
            t_req = self.env.now
            yield req
            row.token_start = self.env.now
            row.wait_token = row.token_start - t_req

            # Service
            st = float(row.token_service_min)
            self.busy_token += st
            yield from self._serve(st)

            row.token_end = self.env.now

        # ---- Stage 2 arrival ----
        row.stage2_arrival = row.token_end

        # ---- Stage 2: Two parallel tandoors, one common queue (FIFO by SimPy request order) ----
        # We implement “whoever is free first takes the next token” by requesting both and taking whichever grants first.
        # NOTE: This preserves FIFO at the shared-queue level.
        req1 = self.t1.request()
        req2 = self.t2.request()

        t_req2 = self.env.now
        res = yield req1 | req2  # whichever becomes available first

        if req1 in res and req2 in res:
            # extremely rare; pick t1
            chosen = 1
            self.t2.release(req2)
        elif req1 in res:
            chosen = 1
            self.t2.release(req2)
        else:
            chosen = 2
            self.t1.release(req1)

        row.assigned_tandoor = chosen
        row.tandoor_start = self.env.now
        row.wait_tandoor = row.tandoor_start - t_req2

        st2 = float(row.tandoor_service_min)
        if chosen == 1:
            self.busy_t1 += st2
        else:
            self.busy_t2 += st2

        yield from self._serve(st2)
        row.tandoor_end = self.env.now

        # Release chosen
        if chosen == 1:
            self.t1.release(req1)
        else:
            self.t2.release(req2)


def run_simulation(
    input_csv: str,
    hide_assigned_tandoor_in_export: bool = True,
    export_prefix: str = "OUTPUT"
) -> Dict[str, Any]:
    df = pd.read_csv(input_csv)

    # Standardize token number
    if "Token_No" in df.columns:
        token_no = df["Token_No"].astype(int).to_numpy()
    elif "Customer_ID" in df.columns:
        token_no = df["Customer_ID"].astype(int).to_numpy()
    else:
        token_no = np.arange(1, len(df) + 1)

    df = ensure_arrival_minutes(df)

    # Interarrival
    if "InterArrival_min" in df.columns:
        inter = pd.to_numeric(df["InterArrival_min"], errors="coerce").fillna(0).to_numpy()
    else:
        arr = df["Arrival_min"].to_numpy()
        inter = np.zeros_like(arr)
        inter[1:] = np.maximum(0.0, arr[1:] - arr[:-1])

    # Service times
    if "TokenService_min" not in df.columns or "TandoorService_min" not in df.columns:
        raise ValueError("Dataset must include TokenService_min and TandoorService_min.")

    token_service = pd.to_numeric(df["TokenService_min"], errors="coerce").to_numpy()
    tandoor_service = pd.to_numeric(df["TandoorService_min"], errors="coerce").to_numpy()

    # Arrival minutes
    arrival_min = pd.to_numeric(df["Arrival_min"], errors="coerce").to_numpy()

    # Start clock label (for readable time)
    start_clock = df["Arrival_24h"].iloc[0] if "Arrival_24h" in df.columns else "19:00:00"

    # Build logs
    logs = [
        CustomerLog(
            token_no=int(token_no[i]),
            arrival_min=float(arrival_min[i]),
            interarrival_min=float(inter[i]),
            token_service_min=float(token_service[i]),
            tandoor_service_min=float(tandoor_service[i]),
        )
        for i in range(len(df))
    ]

    # Run DES
    env = simpy.Environment()
    model = TwoStageTandoorDES(env)

    for row in logs:
        env.process(model.customer_process(row))

    env.run()  # runs until all processes finish
    makespan = env.now

    # Output dataframe
    out = pd.DataFrame([{
        "Token_No": r.token_no,
        "Arrival_min": r.arrival_min,
        "Arrival_24h": clock_from_start(str(start_clock), r.arrival_min),
        "InterArrival_min": r.interarrival_min,

        "TokenService_min": r.token_service_min,
        "Token_Start_min": r.token_start,
        "Token_End_min": r.token_end,
        "Token_Start_24h": clock_from_start(str(start_clock), r.token_start),
        "Token_End_24h": clock_from_start(str(start_clock), r.token_end),
        "Wait_Token_min": r.wait_token,

        "Stage2_Arrival_min": r.stage2_arrival,
        "Stage2_Arrival_24h": clock_from_start(str(start_clock), r.stage2_arrival),

        "Assigned_Tandoor": r.assigned_tandoor,
        "TandoorService_min": r.tandoor_service_min,
        "Tandoor_Start_min": r.tandoor_start,
        "Tandoor_End_min": r.tandoor_end,
        "Tandoor_Start_24h": clock_from_start(str(start_clock), r.tandoor_start),
        "Tandoor_End_24h": clock_from_start(str(start_clock), r.tandoor_end),
        "Wait_Tandoor_min": r.wait_tandoor,

        "Total_System_Time_min": (r.tandoor_end - r.arrival_min),
    } for r in logs])

    # KPIs
    kpi = {
        "N": len(out),
        "Makespan_min": makespan,
        "Avg_Wait_Token_min": float(out["Wait_Token_min"].mean()),
        "Avg_Wait_Tandoor_min": float(out["Wait_Tandoor_min"].mean()),
        "Avg_Total_System_Time_min": float(out["Total_System_Time_min"].mean()),
        "Token_Utilization": float(model.busy_token / makespan) if makespan > 0 else np.nan,
        "Tandoor1_Utilization": float(model.busy_t1 / makespan) if makespan > 0 else np.nan,
        "Tandoor2_Utilization": float(model.busy_t2 / makespan) if makespan > 0 else np.nan,
        "Throughput_per_min": float(len(out) / makespan) if makespan > 0 else np.nan,
    }

    # Hide Assigned_Tandoor if user wants it removed in export
    export_out = out.drop(columns=["Assigned_Tandoor"]) if hide_assigned_tandoor_in_export else out

    csv_out = f"{export_prefix}_tandoor_results.csv"
    xlsx_out = f"{export_prefix}_tandoor_results.xlsx"
    export_out.to_csv(csv_out, index=False)
    export_out.to_excel(xlsx_out, index=False)

    # KPI file
    kpi_df = pd.DataFrame([kpi])
    kpi_csv = f"{export_prefix}_kpi.csv"
    kpi_df.to_csv(kpi_csv, index=False)

    return {
        "results_df": export_out,
        "kpi": kpi,
        "results_csv": csv_out,
        "results_xlsx": xlsx_out,
        "kpi_csv": kpi_csv,
    }


if __name__ == "__main__":
    # Change this to your dataset path
    INPUT = "tandoor_simulator_19_to_21_simple.csv"

    r = run_simulation(
        input_csv=INPUT,
        hide_assigned_tandoor_in_export=True,  # you asked to hide it
        export_prefix="EVENING"
    )

    print("DONE. Files created:")
    print(" -", r["results_csv"])
    print(" -", r["results_xlsx"])
    print(" -", r["kpi_csv"])
    print("\nKPIs:")
    for k, v in r["kpi"].items():
        print(f"{k}: {v}")
