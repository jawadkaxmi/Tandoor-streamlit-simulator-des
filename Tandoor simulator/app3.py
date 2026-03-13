# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from ui.theme import inject_theme
from ui.components import section, end_section, model_gallery
from core.models import parse_label, build_arrival_times, build_service_times
from core.des import simulate_tandem_two_stage


# =========================================================
# Robust Chi-square GOF (Exponential) — FULL WORKING
# Uses equiprobable bins => Expected counts are equal => stable
# =========================================================
def chi_square_exponential_gof_full(x: np.ndarray, bins: int = 10, alpha: float = 0.05) -> dict:
    """
    Chi-square GOF for Exponential.
    - Fits lambda_hat = 1/mean(x)
    - Builds equiprobable bins (quantiles) so Expected = n/k each bin
    - df = (k-1) - 1 (because 1 parameter estimated)
    """
    try:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        x = x[x > 0]  # exponential support

        n = int(len(x))
        if n < 20:
            return {"ok": False, "error": f"Not enough valid samples for Chi-square (n={n}). Need >= 20."}

        mean_x = float(np.mean(x))
        if mean_x <= 0:
            return {"ok": False, "error": "Mean is <= 0, cannot fit exponential."}

        lambda_hat = 1.0 / mean_x

        # choose k so expected >= 5
        k = int(min(bins, max(3, n // 5)))
        # quantile edges for equiprobable bins
        ps = np.linspace(0, 1, k + 1)
        edges = np.zeros_like(ps)

        # Exponential quantile: Q(p)= -ln(1-p)/lambda
        # Use finite cap for p=1 => inf
        for i, p in enumerate(ps):
            if p <= 0:
                edges[i] = 0.0
            elif p >= 1:
                edges[i] = np.inf
            else:
                edges[i] = -np.log(1 - p) / lambda_hat

        # observed counts
        observed, _ = np.histogram(x, bins=edges)

        expected_each = n / k
        expected = np.full(k, expected_each, dtype=float)

        chi_components = (observed - expected) ** 2 / expected
        chi_square = float(np.sum(chi_components))

        # df adjustment: bins-1 - params_estimated(=1)
        df = max(1, (k - 1) - 1)

        # p-value + critical using scipy if available
        try:
            from scipy.stats import chi2
            p_value = float(chi2.sf(chi_square, df))
            critical = float(chi2.ppf(1 - alpha, df))
        except Exception:
            p_value = float("nan")
            critical = float("nan")

        decision = "Fail to reject H0 (Exponential)" if (not np.isnan(p_value) and p_value >= alpha) else "Reject H0 (Not Exponential)"

        # build table
        rows = []
        for i in range(k):
            lower = float(edges[i])
            upper = float(edges[i + 1]) if np.isfinite(edges[i + 1]) else np.inf
            rows.append({
                "Bin": i + 1,
                "Lower": lower,
                "Upper": upper,
                "Observed": int(observed[i]),
                "Expected": float(expected[i]),
                "Chi_Component": float(chi_components[i]),
            })
        table = pd.DataFrame(rows)

        return {
            "ok": True,
            "n": n,
            "mean": mean_x,
            "lambda_hat": lambda_hat,
            "bins": k,
            "chi_square": chi_square,
            "df": df,
            "alpha": alpha,
            "critical": critical,
            "p_value": p_value,
            "decision": decision,
            "table": table,
        }

    except Exception as e:
        return {"ok": False, "error": f"Chi-square failed: {e}"}


# =========================================================
# UI + App
# =========================================================
inject_theme()
st.markdown("<div class='wrap'>", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="logo"></div>
    <div class="title">
      <h1>Tandoor Queueing Lab</h1>
      <p>Two-stage DES • Token Counter → Tandoor (parallel servers)</p>
    </div>
  </div>
  <div class="pill"><b>Modern Web Simulator</b> • Data-driven / Rate-based</div>
</div>
""", unsafe_allow_html=True)

if "sim" not in st.session_state:
    st.session_state["sim"] = None
if "chi" not in st.session_state:
    st.session_state["chi"] = None
if "params" not in st.session_state:
    st.session_state["params"] = None


# ----------------------------
# Layout
# ----------------------------
left, right = st.columns([1.0, 1.2], gap="large")

with left:
    c = section("1) Input", "Upload final dataset OR run a parametric scenario.")
    mode = st.radio("Mode", ["Data-driven (CSV)", "Rate-based (λ, μ)"], horizontal=True)

    uploaded = st.file_uploader("Upload FINAL CSV dataset", type=["csv"], disabled=(mode != "Data-driven (CSV)"))
    st.caption("Final CSV required columns: Arrival_Time (HH:MM:SS), Token_Service_Time, Tandoor_Service_Time")

    st.markdown("**Rate-based Parameters (per minute)**")
    lam_in = st.number_input("λ (arrival rate / min)", 0.01, 60.0, 4.00, 0.10, disabled=(mode != "Rate-based (λ, μ)"))
    mu1_in = st.number_input("μ₁ Token service rate / min", 0.01, 60.0, 2.00, 0.10, disabled=(mode != "Rate-based (λ, μ)"))
    mu2_in = st.number_input("μ₂ Tandoor service rate / min (per server)", 0.01, 60.0, 0.50, 0.10, disabled=(mode != "Rate-based (λ, μ)"))

    end_section()

    c = section("2) Configuration", "Simulation size & servers.")
    N = st.number_input("Customers (N)", 10, 5000, 150, 10)
    seed = st.number_input("Random seed", 0, 9_999_999, 42, 1)
    default_c_stage2 = st.number_input("If model uses 'c': tandoors (c)", 1, 20, 2, 1)
    hide_tandoor_server = st.toggle("Hide which tandoor served the customer in export", value=True)

    colA, colB = st.columns(2)
    run = colA.button("Run Simulation", type="primary", use_container_width=True)
    if colB.button("Clear Results", use_container_width=True):
        st.session_state["sim"] = None
        st.session_state["chi"] = None
        st.session_state["params"] = None
        st.rerun()
    end_section()

with right:
    c = section("3) Queueing Model Selection", "Choose a Kendall model for each stage (selection affects logic).")
    st.markdown("### Stage 1 (Token Counter)")
    sel1 = model_gallery("model_stage1", default_label="G/G/1")
    st.markdown("### Stage 2 (Tandoor Area)")
    sel2 = model_gallery("model_stage2", default_label="G/G/2")
    end_section()


# ----------------------------
# Helpers: FINAL CSV parsing
# ----------------------------
def _load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df

def _parse_arrival_minutes_from_clock(df: pd.DataFrame) -> np.ndarray:
    if "Arrival_Time" not in df.columns:
        raise ValueError("Missing required column: Arrival_Time (HH:MM:SS)")

    # Convert to seconds-of-day
    t = pd.to_datetime(df["Arrival_Time"].astype(str), format="%H:%M:%S", errors="coerce")
    if t.isna().any():
        raise ValueError("Arrival_Time has invalid values. Must be HH:MM:SS.")

    secs = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy(dtype=int)
    # base at first arrival to get minutes-from-start
    base = int(secs.min())
    mins = (secs - base) / 60.0
    mins = mins.astype(float)

    # Ensure nondecreasing for simulation: sort (FIFO arrival order)
    return np.sort(mins)

def _get_empirical_final(df: pd.DataFrame):
    arr = _parse_arrival_minutes_from_clock(df)

    if "Token_Service_Time" not in df.columns:
        raise ValueError("Missing required column: Token_Service_Time")
    if "Tandoor_Service_Time" not in df.columns:
        raise ValueError("Missing required column: Tandoor_Service_Time")

    tok = pd.to_numeric(df["Token_Service_Time"], errors="coerce").to_numpy(dtype=float)
    tan = pd.to_numeric(df["Tandoor_Service_Time"], errors="coerce").to_numpy(dtype=float)

    tok = tok[np.isfinite(tok) & (tok > 0)]
    tan = tan[np.isfinite(tan) & (tan > 0)]
    return arr, tok, tan


# ----------------------------
# Run simulation
# ----------------------------
if run:
    try:
        sel1_label = sel1
        sel2_label = sel2
        model1 = parse_label(sel1)
        model2 = parse_label(sel2)
        rng = np.random.default_rng(int(seed))

        if mode == "Data-driven (CSV)":
            if uploaded is None:
                st.error("Upload the FINAL CSV to run Data-driven mode.")
                st.stop()

            raw = _load_csv(uploaded)
            emp_arr, emp_tok, emp_tan = _get_empirical_final(raw)

            # --- Parameter estimation from data ---
            inter_emp = np.diff(emp_arr)
            inter_emp = inter_emp[np.isfinite(inter_emp) & (inter_emp > 0)]

            if len(inter_emp) < 2:
                raise ValueError("Not enough valid inter-arrivals after cleaning.")

            mean_ia = float(np.mean(inter_emp))
            mean_tok = float(np.mean(emp_tok)) if len(emp_tok) else 0.5
            mean_tan = float(np.mean(emp_tan)) if len(emp_tan) else 1.0

            lam_hat = 1.0 / mean_ia
            mu1_hat = 1.0 / mean_tok
            mu2_hat = 1.0 / mean_tan

            st.session_state["params"] = {
                "lambda": lam_hat,
                "mu1": mu1_hat,
                "mu2": mu2_hat,
                "mean_inter": mean_ia,
                "mean_tok": mean_tok,
                "mean_tan": mean_tan
            }

            # Build simulation inputs according to selected Kendall models
            arr1 = build_arrival_times(
                model1, int(N), rng,
                mean_interarrival=mean_ia,
                empirical_arrival_times=emp_arr
            )
            svc1 = build_service_times(
                model1, int(N), rng,
                mean_service=mean_tok,
                empirical_service_times=emp_tok
            )
            svc2 = build_service_times(
                model2, int(N), rng,
                mean_service=mean_tan,
                empirical_service_times=emp_tan
            )

            # Chi-square GOF (run ONLY where we claim 'M' / Exponential)
            # For MG1 + GG2 typical use:
            #  - Arrivals: test inter-arrival exponential (supports 'M' for Stage-1 arrival)
            #  - Token service: treat as General if you selected 'G' (skip GOF)
            #  - Tandoor service: treat as General if you selected 'G' (skip GOF)

            def _is_M_arrival(label: str) -> bool:
                # label like 'M/G/1'
                return str(label).strip().upper().startswith("M/")
            def _is_M_service(label: str) -> bool:
                # service letter is the middle part: 'M/G/1' -> 'G'
                parts = str(label).strip().upper().split("/")
                return len(parts) >= 2 and parts[1] == "M"

            chi_arr = chi_square_exponential_gof_full(inter_emp, bins=10, alpha=0.05) if _is_M_arrival(sel1_label) else {"ok": False, "error": "Arrival not assumed exponential ('M') for Stage 1; Chi-square skipped."}

            chi_tok = chi_square_exponential_gof_full(emp_tok, bins=10, alpha=0.05) if _is_M_service(sel1_label) else {"ok": False, "error": "Token service treated as General ('G'); Chi-square skipped."}

            chi_tan = chi_square_exponential_gof_full(emp_tan, bins=10, alpha=0.05) if _is_M_service(sel2_label) else {"ok": False, "error": "Tandoor service treated as General ('G'); Chi-square skipped."}

            st.session_state["chi"] = {"arrivals": chi_arr, "token": chi_tok, "tandoor": chi_tan}

        else:
            # --- Rate-based inputs (λ, μ) ---
            lam = float(lam_in)
            mu1 = float(mu1_in)
            mu2 = float(mu2_in)

            mean_ia = 1.0 / lam
            mean_tok = 1.0 / mu1
            mean_tan = 1.0 / mu2

            st.session_state["params"] = {
                "lambda": lam,
                "mu1": mu1,
                "mu2": mu2,
                "mean_inter": mean_ia,
                "mean_tok": mean_tok,
                "mean_tan": mean_tan
            }

            # For M: generate exponential baseline
            base_arr = np.cumsum(np.r_[0.0, rng.exponential(scale=mean_ia, size=int(N) - 1)])
            base_tok = rng.exponential(scale=mean_tok, size=int(N))
            base_tan = rng.exponential(scale=mean_tan, size=int(N))

            arr1 = build_arrival_times(model1, int(N), rng, mean_interarrival=mean_ia, empirical_arrival_times=base_arr)
            svc1 = build_service_times(model1, int(N), rng, mean_service=mean_tok, empirical_service_times=base_tok)
            svc2 = build_service_times(model2, int(N), rng, mean_service=mean_tan, empirical_service_times=base_tan)

            # Chi-square GOF (run ONLY where we claim 'M' / Exponential)
            inter_gen = np.diff(arr1)
            inter_gen = inter_gen[np.isfinite(inter_gen) & (inter_gen > 0)]

            def _is_M_arrival(label: str) -> bool:
                return str(label).strip().upper().startswith("M/")
            def _is_M_service(label: str) -> bool:
                parts = str(label).strip().upper().split("/")
                return len(parts) >= 2 and parts[1] == "M"

            chi_arr = chi_square_exponential_gof_full(inter_gen, bins=10, alpha=0.05) if _is_M_arrival(sel1_label) else {"ok": False, "error": "Arrival not assumed exponential ('M') for Stage 1; Chi-square skipped."}
            chi_tok = chi_square_exponential_gof_full(svc1, bins=10, alpha=0.05) if _is_M_service(sel1_label) else {"ok": False, "error": "Token service treated as General ('G'); Chi-square skipped."}
            chi_tan = chi_square_exponential_gof_full(svc2, bins=10, alpha=0.05) if _is_M_service(sel2_label) else {"ok": False, "error": "Tandoor service treated as General ('G'); Chi-square skipped."}

            st.session_state["chi"] = {"arrivals": chi_arr, "token": chi_tok, "tandoor": chi_tan}

        sim = simulate_tandem_two_stage(
            arrival_times_stage1=arr1,
            service_times_stage1=svc1,
            service_times_stage2=svc2,
            model_stage1=model1,
            model_stage2=model2,
            default_c_stage2=int(default_c_stage2),
        )

        st.session_state["sim"] = sim
        st.success("Simulation completed.")

    except Exception as e:
        st.error(str(e))


# ----------------------------
# Display
# ----------------------------
sim = st.session_state.get("sim")

if not isinstance(sim, dict):
    st.info("Run a simulation to see results.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# --- Parameter Estimation / Inputs (NO CV ANYWHERE) ---
params = st.session_state.get("params") or {}
st.markdown("## Parameter Estimation / Inputs")

if params:
    a, b, c = st.columns(3)
    a.metric("λ (arrival rate / min)", f"{params['lambda']:.6f}")
    b.metric("μ₁ (token service / min)", f"{params['mu1']:.6f}")
    c.metric("μ₂ (tandoor service / min per server)", f"{params['mu2']:.6f}")

    a2, b2, c2 = st.columns(3)
    a2.metric("Mean Inter-arrival (min)", f"{params['mean_inter']:.4f}")
    b2.metric("Mean Token Service (min)", f"{params['mean_tok']:.4f}")
    c2.metric("Mean Tandoor Service (min)", f"{params['mean_tan']:.4f}")


# --- Chi-square section (FULL TABLES) ---
chi = st.session_state.get("chi")
if chi is not None:
    st.markdown("## Statistical Validation (Chi-square — Full Working)")

    def show_chi(title: str, res: dict):
        st.markdown(f"### {title}")
        if not res.get("ok"):
            st.warning(res.get("error", "Chi-square failed."))
            return
        st.write(
            f"n={res['n']} | mean={res['mean']:.6f} | λ̂={res['lambda_hat']:.6f}"
        )
        st.write(
            f"χ²={res['chi_square']:.4f} | df={res['df']} | α={res['alpha']} | "
            f"Critical={res['critical']:.4f} | p-value={res['p_value']:.6f}"
        )
        st.write(f"Decision: **{res['decision']}**")
        st.dataframe(res["table"], use_container_width=True)

    show_chi("Arrivals (Inter-arrival)", chi["arrivals"])
    show_chi("Token Service (Stage 1)", chi["token"])
    show_chi("Tandoor Service (Stage 2)", chi["tandoor"])


# --- KPIs ---
overall = sim["overall"]
combined = sim["combined"].copy()

st.markdown("## Results")

st.markdown("""
<div class="kpis">
  <div class="kpi"><div class="label">Stage 1 Model</div><div class="value">{}</div><div class="delta">Servers: {}</div></div>
  <div class="kpi"><div class="label">Stage 2 Model</div><div class="value">{}</div><div class="delta">Servers: {}</div></div>
  <div class="kpi"><div class="label">Avg Wait (Token)</div><div class="value">{:.3f} min</div><div class="delta">Stage 1 queue</div></div>
  <div class="kpi"><div class="label">Avg Wait (Tandoor)</div><div class="value">{:.3f} min</div><div class="delta">Stage 2 queue</div></div>
</div>
""".format(
    overall["model_stage1"], overall["token_servers"],
    overall["model_stage2"], overall["tandoor_servers"],
    overall["avg_wait_token"], overall["avg_wait_tandoor"]
), unsafe_allow_html=True)

st.markdown("""
<div class="kpis" style="margin-top:12px;">
  <div class="kpi"><div class="label">Avg Total System Time</div><div class="value">{:.3f} min</div><div class="delta">End-to-end</div></div>
  <div class="kpi"><div class="label">Customers</div><div class="value">{}</div><div class="delta">Simulated</div></div>
  <div class="kpi"><div class="label">Stage 2 Makespan</div><div class="value">{:.2f} min</div><div class="delta">Last completion</div></div>
  <div class="kpi"><div class="label">Tandoor Utilization</div><div class="value">{}</div><div class="delta">Per server</div></div>
</div>
""".format(
    overall["avg_total_system_time"], overall["n"], overall["makespan_stage2"],
    ", ".join([f"S{k}:{v:.2f}" for k, v in overall["tandoor_utilization"].items()])
), unsafe_allow_html=True)


# --- Tabs: tables + charts ---
tab1, tab2, tab3, tab4 = st.tabs(["Combined Table", "Event Tables", "Charts", "Gantt"])

with tab1:
    view_df = combined.copy()
    if hide_tandoor_server and "Tandoor_Server" in view_df.columns:
        view_df = view_df.drop(columns=["Tandoor_Server"])
    st.dataframe(view_df, use_container_width=True, height=420)

    out = io.StringIO()
    view_df.to_csv(out, index=False)
    st.download_button("Download results CSV", data=out.getvalue(), file_name="tandoor_two_stage_results.csv", mime="text/csv")

with tab2:
    st.markdown("### Stage 1 Events (Token Counter)")
    st.dataframe(sim["stage1"]["events"], use_container_width=True, height=260)

    st.markdown("### Stage 2 Events (Tandoor Area)")
    st.dataframe(sim["stage2"]["events"], use_container_width=True, height=260)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(combined, x="Wait_Token", nbins=30, title="Wait Token (Stage 1)"), use_container_width=True)
        st.plotly_chart(px.histogram(combined, x="Wait_Tandoor", nbins=30, title="Wait Tandoor (Stage 2)"), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(combined, x="Total_System_Time", nbins=30, title="Total System Time"), use_container_width=True)
        st.plotly_chart(px.line(sim["stage2"]["queue_ts"], x="time", y="queue_length", title="Stage 2 Queue Length over Time"), use_container_width=True)

    st.plotly_chart(px.line(sim["stage1"]["queue_ts"], x="time", y="queue_length", title="Stage 1 Queue Length over Time"), use_container_width=True)

with tab4:
    st.markdown("### Tandoor Busy Timeline (Gantt-style)")
    tl2 = sim["stage2"]["timeline"].copy()

    if tl2.empty:
        st.info("No timeline data.")
    else:
        fig = go.Figure()
        for _, r in tl2.iterrows():
            fig.add_trace(go.Bar(
                x=[float(r["BusyDuration"])],
                y=[f"Tandoor {int(r['Server_ID'])}"],
                base=[float(r["BusyStart"])],
                orientation="h",
                customdata=[[int(r["Customer_ID"])]],
                # IMPORTANT: Plotly placeholder %{base} is not Python variable
                hovertemplate="Customer %{customdata[0]}<br>Start %{base:.3f}<br>Dur %{x:.3f}<extra></extra>",
                showlegend=False
            ))

        fig.update_layout(
            barmode="stack",
            xaxis_title="Time (minutes)",
            yaxis_title="Server",
            height=380
        )

        st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)