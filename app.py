"""Streamlit front end for the Mini Emergency Exit Planner."""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms.astar import compare_searches
from config import BASELINE, DEFAULT_PRIOR, HEAT_FPR, HEAT_TPR, SMOKE_FPR, SMOKE_TPR, STRESS
from pipeline import run_pipeline

st.set_page_config(page_title="Emergency Exit Planner", page_icon="🚪", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1400px}.map{border-collapse:separate;border-spacing:5px;margin:.4rem auto}
.map td{width:48px;height:42px;text-align:center;border-radius:9px;font-weight:700;font-size:15px;color:#17212b}
.legend{font-size:.88rem;line-height:1.9}.muted{color:#667085;font-size:.86rem}
</style>""", unsafe_allow_html=True)

st.title("🚪 Mini Emergency Exit Planner")
st.caption("A* routing + Bayesian sensor reasoning + a one-step MDP decision · educational simulation")
st.warning("Teaching model only. Do not use this demo for real evacuation, fire response, or life-safety decisions.")

def set_preset(sensor, prior, tpr, fpr):
    st.session_state["sensor"] = sensor
    st.session_state["prior"] = prior
    st.session_state["tpr"] = tpr
    st.session_state["fpr"] = fpr
    st.session_state["risk"] = 100


if "sensor" not in st.session_state:
    set_preset("alert", DEFAULT_PRIOR, SMOKE_TPR, SMOKE_FPR)
if "walls" not in st.session_state:
    st.session_state.walls = set(BASELINE.walls)

controls, analysis = st.columns([1.03, 1.2], gap="large")
with controls:
    st.subheader("Scenario and floorplan")
    b1, b2, b3 = st.columns(3)
    b1.button("Clear", on_click=set_preset, args=("clear", DEFAULT_PRIOR, SMOKE_TPR, SMOKE_FPR), use_container_width=True)
    b2.button("Smoke alert", on_click=set_preset, args=("alert", DEFAULT_PRIOR, SMOKE_TPR, SMOKE_FPR), use_container_width=True)
    b3.button("Noisy sensor", on_click=set_preset, args=("alert", DEFAULT_PRIOR, 0.60, 0.40), use_container_width=True)

    grid_choice = st.radio("Grid", ["6 x 6 baseline", "10 x 10 stress test"], horizontal=True)
    building = BASELINE if grid_choice.startswith("6") else STRESS
    if grid_choice.startswith("10"):
        walls = set(STRESS.walls)
    else:
        walls = set(st.session_state.walls)

    prior = st.slider("Prior probability corridor is blocked", 0.0, 1.0, float(st.session_state.prior), 0.01, key="prior")
    tpr = st.slider("Smoke sensor true positive rate", 0.0, 1.0, float(st.session_state.tpr), 0.01, key="tpr")
    fpr = st.slider("Smoke sensor false positive rate", 0.0, 1.0, float(st.session_state.fpr), 0.01, key="fpr")
    sensor = st.radio("Smoke sensor observation", ["clear", "alert"], horizontal=True, key="sensor")
    risk = st.slider("Penalty if proceeding into a blocked corridor", 0, 300, int(st.session_state.risk), 5, key="risk")
    use_heat = st.checkbox("Include a second, conditionally independent heat sensor")
    heat_observation = st.radio("Heat sensor observation", ["clear", "alert"], horizontal=True, disabled=not use_heat) if use_heat else None

    if grid_choice.startswith("6"):
        protected = {building.start, building.exit_a, building.exit_b}
        editable = [(r, c) for r in range(building.size) for c in range(building.size) if (r, c) not in protected]
        labels = [f"row {r + 1}, column {c + 1}" for r, c in editable]
        selected_label = st.selectbox("Edit a wall cell (optional)", labels, key="wall_cell")
        selected_cell = editable[labels.index(selected_label)]
        add_wall = selected_cell not in walls
        if st.button(("Add wall" if add_wall else "Remove wall") + f" at {selected_label}"):
            if add_wall:
                st.session_state.walls.add(selected_cell)
            else:
                st.session_state.walls.discard(selected_cell)
            st.rerun()

    result = run_pipeline(building, sensor, prior, tpr, fpr, risk,
                          heat_observation=heat_observation, walls=walls)
    if result.route.path is None:
        st.error(f"No safe path to {result.selected_exit} on this map. Edit the walls or choose another grid.")
    st.markdown("<div class='legend'>🟢 Start &nbsp; 🟥 Wall &nbsp; 🚪 Exit A &nbsp; 🛟 Exit B &nbsp; 🟠 Sensor cell &nbsp; 🟦 Route &nbsp; · explored node</div>", unsafe_allow_html=True)

    def render_map(agent=None):
        route = set(result.route.path or [])
        explored = set(result.route.expanded)
        rows = []
        for r in range(building.size):
            cells = []
            for c in range(building.size):
                cell = (r, c)
                label, color = "", "#f2f4f7"
                if cell == building.start:
                    label, color = "S", "#a6f4c5"
                elif cell == building.exit_a:
                    label, color = "A", "#fecaca"
                elif cell == building.exit_b:
                    label, color = "B", "#c4b5fd"
                elif cell in walls or cell in result.effective_walls:
                    label, color = "×", "#475467"
                elif agent == cell:
                    label, color = "●", "#facc15"
                elif cell == building.monitored_corridor:
                    label, color = "◉", "#fed7aa"
                elif cell in route:
                    label, color = "•", "#93c5fd"
                elif cell in explored:
                    label, color = "·", "#dbeafe"
                cells.append(f"<td style='background:{color}' title='{cell}'>{label}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        return "<table class='map'>" + "".join(rows) + "</table>"

    board = st.empty()
    board.markdown(render_map(), unsafe_allow_html=True)
    if st.button("▶ Simulate evacuation", type="primary", disabled=result.route.path is None):
        for cell in result.route.path:
            board.markdown(render_map(cell), unsafe_allow_html=True)
            time.sleep(0.12)
        st.success(f"Simulation reached {result.selected_exit} in {result.route.cost} steps.")

with analysis:
    st.subheader("AI decision trace")
    bayes = result.bayes
    with st.expander("1 · A* search", expanded=True):
        st.write(f"**Input:** {building.size}×{building.size} grid, start {building.start}, Exit A {building.exit_a}; 4-way moves; unit cost 1.")
        st.latex(r"f(n)=g(n)+h(n),\quad h(n)=|r-r_g|+|c-c_g|")
        st.write(f"**Initial route:** {result.initial_route.cost} steps; {result.initial_route.expanded_count} nodes expanded. **Selected route:** {result.route.cost} steps to {result.selected_exit}.")
        st.write("Manhattan distance is admissible for four-direction unit-cost movement, so A* returns a shortest path.")
        if result.route.trace:
            st.dataframe(pd.DataFrame(result.route.trace[:12]), hide_index=True, use_container_width=True)

    with st.expander("2 · Bayesian sensor update", expanded=True):
        prior_now = prior
        for i, update_result in enumerate(bayes, 1):
            st.write(f"**Sensor {i} ({update_result.observation.upper()}):** prior={prior_now:.4f}, P(E|Blocked)={update_result.likelihood_if_blocked:.4f}, P(E|Clear)={update_result.likelihood_if_clear:.4f}, P(E)={update_result.evidence:.4f} → posterior={update_result.posterior:.4%}.")
            st.latex(r"P(B|E)=\frac{P(E|B)P(B)}{P(E|B)P(B)+P(E|\neg B)P(\neg B)}")
            prior_now = update_result.posterior
        st.caption("The optional heat sensor is treated as conditionally independent of the smoke sensor given corridor state.")

    with st.expander("3 · MDP policy", expanded=True):
        d = result.decision
        st.write(f"**Input:** p={d.blockage_probability:.4%}; rewards: proceed-clear +100, proceed-blocked {-risk:g}, detour +40.")
        st.latex(r"EU(Proceed)=(1-p)R_{clear}+pR_{blocked};\quad EU(Detour)=40")
        st.write(f"**Output:** EU(Proceed)={d.proceed_utility:.2f}, EU(Detour)={d.detour_utility:.2f}, boundary p*={d.threshold:.2%} → **{d.action}** → {result.selected_exit}.")
        st.caption("At exact utility ties the policy chooses the safer detour. This is a one-step toy MDP; movement and changing hazards are not modeled.")

    st.subheader("A* and BFS benchmark")
    comparison = compare_searches(building.size, building.start, building.exit_a, walls)
    st.dataframe(pd.DataFrame(comparison).assign(**{"Runtime (ms)": [round(r["Runtime (ms)"], 4) for r in comparison]}), hide_index=True, use_container_width=True)
    st.caption("Both searches use the same map and goal. Timings are machine-dependent and noisy on tiny maps; compare path length and expanded nodes first.")

    st.subheader("MDP decision boundary")
    ps = [i / 100 for i in range(101)]
    proceed = [(1-p)*100 + p*(-risk) for p in ps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ps, y=proceed, mode="lines", name="EU(Proceed)", line={"color": "#2563eb", "width": 3}))
    fig.add_trace(go.Scatter(x=ps, y=[40]*len(ps), mode="lines", name="EU(Detour)", line={"color": "#ea580c", "width": 3}))
    fig.add_vline(x=d.threshold, line_dash="dash", line_color="#667085", annotation_text=f"boundary {d.threshold:.1%}")
    fig.add_vline(x=d.blockage_probability, line_dash="dot", line_color="#16a34a", annotation_text="current p")
    fig.update_layout(height=330, margin={"l": 10, "r": 10, "t": 25, "b": 10}, xaxis_title="P(corridor blocked)", yaxis_title="Expected utility", legend={"orientation": "h", "y": 1.12})
    st.plotly_chart(fig, use_container_width=True)

st.caption("Assumptions are illustrative. This application does not connect to sensors, validate a real building, or provide life-safety guidance.")
