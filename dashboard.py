"""Gradient Descent Media Optimization — Live Dashboard.

Usage:
    streamlit run dashboard.py
    streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants (mirrored from gradient_descent.py)
# ---------------------------------------------------------------------------
ROWS = ["A", "B", "C", "D", "E", "F", "G", "H"]
COLS = list(range(1, 13))
SUPPLEMENT_NAMES = ["Glucose", "NaCl", "MgSO4"]
WELL_VOLUME_UL = 180
MAX_ITERATIONS = 8

ROW_ROLES = {
    "A": "Control",
    "B": "Center",
    "C": "Glucose +δ",
    "D": "Glucose +δ",
    "E": "NaCl +δ",
    "F": "NaCl +δ",
    "G": "MgSO4 +δ",
    "H": "MgSO4 +δ",
}

ROLE_COLORS = {
    "Control": "#94a3b8",
    "Center": "#3b82f6",
    "Glucose +δ": "#f59e0b",
    "NaCl +δ": "#10b981",
    "MgSO4 +δ": "#8b5cf6",
}

# Reagent source wells on the compound plate
REAGENT_SOURCE = {
    "D1": "Novel_Bio",
    "A1": "Glucose",
    "B1": "NaCl",
    "C1": "MgSO4",
}

DATA_DIR = Path(__file__).parent / "data" / "gradient_descent"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_state() -> dict:
    state_path = DATA_DIR / "state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {
        "current_iteration": 0,
        "current_composition": {"Glucose": 20, "NaCl": 20, "MgSO4": 20},
        "alpha": 1.0,
        "best_od": None,
        "prev_center_od": None,
        "no_improvement_count": 0,
        "converged": False,
        "history": [],
    }


def load_iteration_log(iteration: int) -> dict | None:
    path = DATA_DIR / f"iteration_{iteration}" / "iteration_log.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_transfer_array(iteration: int) -> list | None:
    path = DATA_DIR / f"iteration_{iteration}" / "transfer_array.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_workflow_ids(iteration: int) -> dict | None:
    path = DATA_DIR / f"iteration_{iteration}" / "workflow_ids.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def get_live_workflow_status(workflow_ids: dict) -> dict | None:
    """Try to get live workflow status from workcell MCP."""
    try:
        from gradient_descent import McpClient

        client = McpClient()
        client.connect()
        return client.call_tool(
            "get_workflow_instance_details",
            {"instance_uuid": workflow_ids["workflow_instance_uuid"]},
        )
    except Exception:
        return None


def compute_well_volumes(transfer_array: list) -> dict:
    """Aggregate transfer array into per-well volume breakdowns.

    Returns: {dest_well: {reagent_name: volume, ...}, ...}
    """
    well_volumes = {}
    for src, dest, vol in transfer_array:
        reagent = REAGENT_SOURCE.get(src, src)
        if dest not in well_volumes:
            well_volumes[dest] = {}
        well_volumes[dest][reagent] = well_volumes[dest].get(reagent, 0) + vol
    return well_volumes


# ---------------------------------------------------------------------------
# Plate heatmap
# ---------------------------------------------------------------------------


def build_plate_data(state: dict) -> tuple[np.ndarray, list[list[str]]]:
    """Build 8x12 arrays for OD600 values and hover text."""
    od_grid = np.full((8, 12), np.nan)
    hover_grid = [["" for _ in range(12)] for _ in range(8)]

    for h in state.get("history", []):
        iteration = h["iteration"]
        col_idx = iteration + 1  # column 2 = iter 1, column 3 = iter 2, ...
        if col_idx > 12:
            continue
        col_i = col_idx - 1  # 0-indexed

        od_results = h.get("od_results", {})
        if not od_results:
            log = load_iteration_log(iteration)
            if log:
                od_results = log.get("od_results", {})

        # Load transfer array for volume info
        ta = load_transfer_array(iteration)
        well_vols = compute_well_volumes(ta) if ta else {}

        if od_results:
            control_od = od_results.get("control_od", 0)
            center_od = od_results.get("center_od", 0)
            perturbed = od_results.get("perturbed_ods", {})

            wells_od = {
                "A": control_od,
                "B": center_od,
                "C": perturbed.get("Glucose", [0, 0])[0],
                "D": perturbed.get("Glucose", [0, 0])[1],
                "E": perturbed.get("NaCl", [0, 0])[0],
                "F": perturbed.get("NaCl", [0, 0])[1],
                "G": perturbed.get("MgSO4", [0, 0])[0],
                "H": perturbed.get("MgSO4", [0, 0])[1],
            }

            for row_i, row in enumerate(ROWS):
                od_val = wells_od.get(row, 0)
                od_grid[row_i][col_i] = od_val
                role = ROW_ROLES[row]
                well_name = f"{row}{col_idx}"

                # Build volume breakdown for hover
                vols = well_vols.get(well_name, {})
                vol_lines = "".join(
                    f"{name}: {v} µL<br>"
                    for name, v in sorted(vols.items())
                ) if vols else ""

                hover_grid[row_i][col_i] = (
                    f"<b>{well_name}</b><br>"
                    f"OD600: {od_val:.4f}<br>"
                    f"Role: {role}<br>"
                    f"Iteration {iteration}<br>"
                    f"---<br>"
                    f"{vol_lines}"
                    f"Total: {sum(vols.values()):.0f} µL"
                    if vols else
                    f"<b>{well_name}</b><br>"
                    f"OD600: {od_val:.4f}<br>"
                    f"Role: {role}<br>"
                    f"Iteration {iteration}"
                )

    # Mark seed wells in column 1
    for row_i, row in enumerate(ROWS):
        col_i = 0
        iteration_for_seed = row_i + 1
        if iteration_for_seed <= state.get("current_iteration", 0):
            hover_grid[row_i][col_i] = (
                f"<b>{row}1</b><br>Seed well (round {iteration_for_seed})"
            )
            od_grid[row_i][col_i] = -0.01  # marker value
        else:
            hover_grid[row_i][col_i] = f"<b>{row}1</b><br>Empty seed well"

    # Mark unused columns 10-12
    for col_i in range(9, 12):
        for row_i in range(8):
            hover_grid[row_i][col_i] = f"<b>{ROWS[row_i]}{col_i + 1}</b><br>Unused"

    return od_grid, hover_grid


def plate_heatmap(state: dict) -> go.Figure:
    od_grid, hover_grid = build_plate_data(state)

    fig = go.Figure(
        data=go.Heatmap(
            z=od_grid,
            x=[str(c) for c in COLS],
            y=ROWS,
            text=hover_grid,
            hoverinfo="text",
            colorscale=[
                [0.0, "#dbeafe"],
                [0.05, "#ffffff"],
                [0.5, "#86efac"],
                [1.0, "#15803d"],
            ],
            zmin=-0.02,
            zmax=None,
            colorbar=dict(title="OD600", thickness=15),
            xgap=3,
            ygap=3,
        )
    )

    current_iter = state.get("current_iteration", 0)
    annotations = []
    annotations.append(dict(x="1", y=-0.7, text="Seed", showarrow=False, font=dict(size=10, color="#64748b")))
    for i in range(1, min(current_iter + 1, 9)):
        annotations.append(
            dict(x=str(i + 1), y=-0.7, text=f"Iter {i}", showarrow=False, font=dict(size=10, color="#64748b"))
        )

    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=10, b=40),
        xaxis=dict(title=None, side="top", dtick=1),
        yaxis=dict(title=None, autorange="reversed"),
        annotations=annotations,
        plot_bgcolor="#f8fafc",
    )

    return fig


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def od_progress_chart(history: list) -> go.Figure:
    if not history:
        fig = go.Figure()
        fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
        fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    iters = [h["iteration"] for h in history]
    center = [h.get("center_od", 0) for h in history]
    control = [h.get("control_od", 0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=control, mode="lines+markers", name="Control", line=dict(color="#94a3b8", dash="dash"), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=iters, y=center, mode="lines+markers", name="Center", line=dict(color="#3b82f6", width=3), marker=dict(size=8)))

    # Perturbation means
    colors = {"Glucose": "#f59e0b", "NaCl": "#10b981", "MgSO4": "#8b5cf6"}
    for name in SUPPLEMENT_NAMES:
        means = []
        for h in history:
            od_results = h.get("od_results", {})
            if not od_results:
                log = load_iteration_log(h["iteration"])
                od_results = log.get("od_results", {}) if log else {}
            reps = od_results.get("perturbed_ods", {}).get(name, [0, 0])
            means.append(sum(reps) / max(len(reps), 1))
        fig.add_trace(go.Scatter(x=iters, y=means, mode="lines+markers", name=f"+d {name}", line=dict(color=colors[name], width=1.5), marker=dict(size=5)))

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(title="Iteration", dtick=1),
        yaxis=dict(title="OD600"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#f8fafc",
    )
    return fig


def composition_trajectory_chart(history: list) -> go.Figure:
    if not history:
        fig = go.Figure()
        fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
        fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    iters = [h["iteration"] for h in history]
    colors = {"Novel_Bio": "#64748b", "Glucose": "#f59e0b", "NaCl": "#10b981", "MgSO4": "#8b5cf6"}

    fig = go.Figure()
    for component in ["Novel_Bio"] + SUPPLEMENT_NAMES:
        vals = [h["composition"].get(component, 0) for h in history]
        fig.add_trace(go.Bar(x=iters, y=vals, name=component, marker_color=colors[component]))

    fig.update_layout(
        barmode="stack",
        height=300,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(title="Iteration", dtick=1),
        yaxis=dict(title="Volume (uL)", range=[0, 200]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#f8fafc",
    )
    return fig


def objective_chart(history: list) -> go.Figure:
    """Single clean line: objective function (center OD600) vs iteration."""
    if not history:
        fig = go.Figure()
        fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
        fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    iters = [h["iteration"] for h in history]
    scores = [h.get("center_od", 0) for h in history]

    # Running best
    best_so_far = []
    current_best = 0
    for s in scores:
        current_best = max(current_best, s)
        best_so_far.append(current_best)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=scores, mode="lines+markers",
        name="Score (Center OD)",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=10, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=iters, y=best_so_far, mode="lines",
        name="Best so far",
        line=dict(color="#15803d", width=2, dash="dot"),
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(title="Trial #", dtick=1, range=[0.5, MAX_ITERATIONS + 0.5]),
        yaxis=dict(title="Objective Score (OD600)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#f8fafc",
    )
    return fig


def composition_bar(composition: dict) -> go.Figure:
    novel_bio = WELL_VOLUME_UL - sum(composition.get(n, 0) for n in SUPPLEMENT_NAMES)
    components = ["Novel_Bio"] + SUPPLEMENT_NAMES
    values = [novel_bio] + [composition.get(n, 0) for n in SUPPLEMENT_NAMES]
    colors = ["#64748b", "#f59e0b", "#10b981", "#8b5cf6"]

    fig = go.Figure(
        go.Bar(
            y=components,
            x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{v} uL" for v in values],
            textposition="inside",
            textfont=dict(color="white", size=14),
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(title="Volume (uL)", range=[0, 200]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#f8fafc",
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

st.set_page_config(page_title="GD Media Optimization", layout="wide", page_icon="🧫")

# Custom CSS
st.markdown(
    """
    <style>
    .status-pill {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    .status-running { background: #dbeafe; color: #1d4ed8; }
    .status-converged { background: #dcfce7; color: #15803d; }
    .status-idle { background: #f1f5f9; color: #64748b; }
    .metric-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #1e293b; }
    .metric-label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .explanation {
        font-size: 13px;
        color: #64748b;
        margin-top: -8px;
        margin-bottom: 12px;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    auto_refresh = st.toggle("Auto-refresh (5s)", value=False)
    st.divider()
    st.caption("Gradient Descent Media Optimization")
    st.caption(f"Data: `{DATA_DIR}`")
    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "Each iteration tests the current media recipe (center) plus "
        "small perturbations of each supplement. The gradient of OD600 growth "
        "tells us which direction to adjust. The optimizer steps toward "
        "higher growth, converging on the best recipe.",
        help="Gradient descent on a 96-well plate"
    )
    st.markdown("**Well layout per column**")
    st.markdown(
        "- **Row A** — Control (Novel_Bio only)\n"
        "- **Row B** — Center point (current best)\n"
        "- **Rows C-D** — +Glucose perturbation\n"
        "- **Rows E-F** — +NaCl perturbation\n"
        "- **Rows G-H** — +MgSO4 perturbation"
    )

# Load data
state = load_state()
current_iter = state.get("current_iteration", 0)
history = state.get("history", [])

# Enrich history with od_results from iteration logs if missing
for h in history:
    if "od_results" not in h or not h["od_results"]:
        log = load_iteration_log(h["iteration"])
        if log and "od_results" in log:
            h["od_results"] = log["od_results"]

# Detect status
workflow_ids = load_workflow_ids(current_iter + 1) if current_iter < MAX_ITERATIONS else None
live_status = None
if workflow_ids:
    live_status = get_live_workflow_status(workflow_ids)

is_running = live_status and live_status.get("status") in ("in_progress", "pending_approval", "approved")
is_converged = state.get("converged", False)

# Header
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("GD Media Optimization")
with col_status:
    if is_running:
        running_iter = current_iter + 1
        st.markdown(f'<div class="status-pill status-running">Running Iteration {running_iter}</div>', unsafe_allow_html=True)
    elif is_converged:
        st.markdown('<div class="status-pill status-converged">Converged</div>', unsafe_allow_html=True)
    else:
        status_text = f"Idle — {current_iter}/{MAX_ITERATIONS} done" if current_iter > 0 else "Not started"
        st.markdown(f'<div class="status-pill status-idle">{status_text}</div>', unsafe_allow_html=True)

# Metrics row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Iteration", f"{current_iter} / {MAX_ITERATIONS}")
with m2:
    best_od = state.get("best_od")
    st.metric("Best Center OD", f"{best_od:.4f}" if best_od else "—")
with m3:
    st.metric("Learning Rate (a)", f"{state.get('alpha', 1.0):.2f}")
with m4:
    streak = state.get("no_improvement_count", 0)
    st.metric("No-improvement streak", f"{streak} / 2")

st.divider()

# Objective function chart — the hero chart
st.subheader("Optimization Score")
st.markdown(
    '<div class="explanation">'
    "The objective function: center-point OD600 at each trial. "
    "If the optimization is working, this line should trend upward and flatten as it converges. "
    "The dotted green line tracks the best score seen so far."
    '</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(objective_chart(history), use_container_width=True, key="objective")

st.divider()

# Row 1: Plate + Composition
col_plate, col_comp = st.columns([3, 2])
with col_plate:
    st.subheader("Plate View")
    st.markdown(
        '<div class="explanation">'
        "96-well plate colored by OD600 (darker green = higher growth). "
        "Column 1 holds seed wells, columns 2-9 are experiment iterations. "
        "Hover over any well to see exact reagent volumes."
        '</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(plate_heatmap(state), use_container_width=True, key="plate")

with col_comp:
    st.subheader("Current Composition")
    st.markdown(
        '<div class="explanation">'
        "The media recipe the optimizer will test next. Each well is 180 uL total — "
        "Novel_Bio is the base media, and the three supplements (Glucose, NaCl, MgSO4) "
        "are the variables being optimized. As supplements increase, Novel_Bio decreases."
        '</div>',
        unsafe_allow_html=True,
    )
    comp = state.get("current_composition", {"Glucose": 20, "NaCl": 20, "MgSO4": 20})
    st.plotly_chart(composition_bar(comp), use_container_width=True, key="comp_bar")

# Row 2: Progress charts
col_od, col_traj = st.columns(2)
with col_od:
    st.subheader("OD600 Progress")
    st.markdown(
        '<div class="explanation">'
        "Tracks bacterial growth (OD600) across iterations. The <b>center</b> line (blue) "
        "is the current best recipe — we want this to go up. The <b>control</b> (gray dashed) "
        "is Novel_Bio only, serving as a baseline. Colored lines show the mean growth when each "
        "supplement was increased. If a colored line is above center, that supplement helps growth."
        '</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(od_progress_chart(history), use_container_width=True, key="od_progress")
with col_traj:
    st.subheader("Composition Trajectory")
    st.markdown(
        '<div class="explanation">'
        "How the media recipe changes over time. Each stacked bar shows the volume breakdown "
        "for that iteration's center point. The optimizer increases supplements with positive "
        "gradients and decreases those with negative gradients, always keeping the total at 180 uL."
        '</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(composition_trajectory_chart(history), use_container_width=True, key="comp_traj")

# Row 3: Current iteration detail
if is_running and live_status:
    st.divider()
    st.subheader(f"Iteration {current_iter + 1} — Live")
    routines = live_status.get("workflow_routines", [])
    completed = sum(1 for r in routines if r.get("status") == "completed")
    total = len(routines)
    st.progress(completed / max(total, 1), text=f"{completed}/{total} routines completed")

    routine_df = pd.DataFrame(
        [{"Routine": r.get("routine_name", "?"), "Status": r.get("status", "?")} for r in routines]
    )
    st.dataframe(routine_df, use_container_width=True, hide_index=True, height=200)

elif history:
    st.divider()
    latest = history[-1]
    latest_log = load_iteration_log(latest["iteration"]) or latest
    gradient = latest.get("gradient", latest_log.get("gradient"))

    if gradient:
        st.subheader(f"Iteration {latest['iteration']} — Gradient")
        st.markdown(
            '<div class="explanation">'
            "Direction the optimizer will move next. An up arrow means increasing that supplement "
            "improved growth; a down arrow means it hurt growth."
            '</div>',
            unsafe_allow_html=True,
        )
        g1, g2, g3 = st.columns(3)
        for col, name in zip([g1, g2, g3], SUPPLEMENT_NAMES):
            val = gradient.get(name, 0)
            arrow = "↑" if val > 0 else ("↓" if val < 0 else "→")
            color = "green" if val > 0 else ("red" if val < 0 else "gray")
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value' style='color:{color}'>{arrow}</div>"
                    f"<div class='metric-label'>{name} ({val:+.4f})</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# History table
if history:
    st.divider()
    st.subheader("Iteration History")
    rows = []
    for h in history:
        c = h.get("composition", {})
        rows.append({
            "Iter": h["iteration"],
            "Novel_Bio": c.get("Novel_Bio", "—"),
            "Glucose": c.get("Glucose", "—"),
            "NaCl": c.get("NaCl", "—"),
            "MgSO4": c.get("MgSO4", "—"),
            "Center OD": f"{h['center_od']:.4f}" if "center_od" in h else "—",
            "Control OD": f"{h['control_od']:.4f}" if "control_od" in h else "—",
            "a": h.get("alpha", "—"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Auto-refresh
if auto_refresh:
    time.sleep(5)
    st.rerun()
