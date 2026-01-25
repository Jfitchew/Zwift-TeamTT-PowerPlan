# ttt_plan_app.py
# Streamlit app to compute a Zwift TTT power/turn plan with fixed rider order (strongest -> weakest),
# constant target speed, drafting CdA reductions, and constraint: each rider's average power over a full
# rotation <= cap_fraction * FTP. Allows weaker riders to do very short or zero pulls.
#
# Run:
#   pip install streamlit pandas numpy
#   streamlit run ttt_plan_app.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

G = 9.80665


# -----------------------------
# Physics / aero helpers
# -----------------------------
def frontal_area_m2(weight_kg: float, height_cm: float) -> float:
    """
    User-provided polynomial.
    IMPORTANT: height in cm (as per user's clarification).
    """
    W = float(weight_kg)
    H = float(height_cm)
    return (
        0.002112 * W
        - 0.00763 * H
        - 0.00000963 * (W ** 2)
        + 0.00001 * W * H
        + 0.0000233 * (H ** 2)
        + 0.7654
    )


def cda_m2(weight_kg: float, height_cm: float, cd: float) -> float:
    return float(cd) * frontal_area_m2(weight_kg, height_cm)


def power_required_w(
    v_mps: float,
    mass_kg: float,
    crr: float,
    rho: float,
    cda_eff: float,
) -> float:
    # Flat, no wind: P = v*(m g Crr) + 0.5*rho*CdA*v^3
    rolling = v_mps * (mass_kg * G * crr)
    aero = 0.5 * rho * cda_eff * (v_mps ** 3)
    return rolling + aero


def build_default_draft_factors(n_riders: int) -> List[float]:
    """
    Your provided draft savings for positions 1..4:
        pos1: 0% reduction => factor 1.00
        pos2: 25% CdA reduction => factor 0.75
        pos3: 30% => 0.70
        pos4: 33% => 0.67
    For n>4 we cap at pos4 factor for all deeper positions by default.
    """
    base = [1.00, 0.75, 0.70, 0.67]
    if n_riders <= 4:
        return base[:n_riders]
    return base + [0.67] * (n_riders - 4)


# -----------------------------
# Rotation model
# -----------------------------
@dataclass
class Rider:
    name: str
    weight_kg: float
    height_cm: float
    ftp_w: float
    bike_kg: float
    cd: float

    @property
    def system_mass_kg(self) -> float:
        return self.weight_kg + self.bike_kg

    @property
    def cda_front(self) -> float:
        return cda_m2(self.weight_kg, self.height_cm, self.cd)


def position_of_rider_during_lead_segment(i: int, k: int, n: int) -> int:
    """
    Segment k means rider k is on the front (position 1).
    Riders are in fixed cyclic order; leader drops to back after their pull.
    During segment k, rider i's position is:
        pos = ((i - k) mod n) + 1   (1-indexed)
    """
    return ((i - k) % n) + 1


def compute_power_matrix(
    riders: List[Rider],
    v_mps: float,
    crr: float,
    rho: float,
    draft_factors: List[float],
) -> np.ndarray:
    """
    Returns P[i,k] = power rider i must do during segment where rider k leads.
    Segment k => rider k at pos1, etc.
    """
    n = len(riders)
    P = np.zeros((n, n), dtype=float)
    for i, r in enumerate(riders):
        for k in range(n):
            pos = position_of_rider_during_lead_segment(i, k, n)
            factor = draft_factors[pos - 1]
            cda_eff = r.cda_front * factor
            P[i, k] = power_required_w(
                v_mps=v_mps,
                mass_kg=r.system_mass_kg,
                crr=crr,
                rho=rho,
                cda_eff=cda_eff,
            )
    return P


def avg_power_for_pulls(P: np.ndarray, pulls_s: np.ndarray) -> np.ndarray:
    """
    Average power per rider over a full rotation:
        avg_i = sum_k pulls[k] * P[i,k] / sum_k pulls[k]
    """
    T = pulls_s.sum()
    if T <= 1e-9:
        return np.zeros(P.shape[0], dtype=float)
    return (P @ pulls_s) / T


# -----------------------------
# Feasibility / pull-time heuristic solver
# -----------------------------
def try_find_feasible_pulls(
    riders: List[Rider],
    v_mps: float,
    crr: float,
    rho: float,
    draft_factors: List[float],
    cap_fraction: float,
    strongest_pull_bounds: Tuple[float, float],
    other_pull_max_s: float,
    min_pull_s: float,
    allow_zero_pull: bool,
    max_iters: int = 2000,
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """
    Heuristic adjustment:
    - Start with strongest pull at midpoint of bounds
    - Others start small and proportional to FTP (but can be 0 if allowed)
    - Iteratively reduce pulls for riders exceeding cap and reassign time to riders with slack
      (prefer strongest/strong riders, limited by their max).
    Returns:
        feasible(bool), pulls_s, avgW, avgFrac
    """
    n = len(riders)
    if n < 2:
        return False, np.zeros(n), np.zeros(n), np.zeros(n)

    P = compute_power_matrix(riders, v_mps, crr, rho, draft_factors)

    # Bounds
    tmin = 0.0 if allow_zero_pull else min_pull_s
    pulls = np.zeros(n, dtype=float)

    # Strongest = index 0 because we will sort by FTP desc.
    pulls[0] = 0.5 * (strongest_pull_bounds[0] + strongest_pull_bounds[1])

    # Initial guess for others: give some work to high FTP riders, less to low FTP.
    ftps = np.array([r.ftp_w for r in riders], dtype=float)
    ftp_norm = (ftps - ftps.min()) / max(1e-9, (ftps.max() - ftps.min()))
    for i in range(1, n):
        # Start small; weaker riders may start at 0 if allowed
        base = 20.0 + 40.0 * ftp_norm[i]  # 20..60s
        pulls[i] = max(tmin, min(base, other_pull_max_s))

    # If zero pulls allowed and we want "strongest to weakest" protection, bias down weakest:
    if allow_zero_pull:
        pulls[-1] = 0.0

    def clamp(i: int):
        if i == 0:
            pulls[i] = float(np.clip(pulls[i], strongest_pull_bounds[0], strongest_pull_bounds[1]))
        else:
            pulls[i] = float(np.clip(pulls[i], tmin, other_pull_max_s))

    for i in range(n):
        clamp(i)

    # Ensure rotation has some total time
    if pulls.sum() < 1e-6:
        return False, pulls, np.zeros(n), np.zeros(n)

    # Iterative reallocation
    for _ in range(max_iters):
        avgW = avg_power_for_pulls(P, pulls)
        caps = cap_fraction * ftps
        frac = avgW / ftps

        # Check feasibility
        if np.all(avgW <= caps + 1e-6):
            return True, pulls, avgW, frac

        # Find most violating rider
        viol = avgW - caps
        i_bad = int(np.argmax(viol))

        # If rider already at minimum pull, cannot reduce further -> infeasible at this speed
        min_allowed = strongest_pull_bounds[0] if i_bad == 0 else tmin
        if pulls[i_bad] <= min_allowed + 1e-9:
            return False, pulls, avgW, frac

        # Remove a small chunk of their pull
        delta = min(2.0, pulls[i_bad] - min_allowed)  # 2s step
        pulls[i_bad] -= delta
        clamp(i_bad)

        # Reassign that time to the rider with most slack (lowest frac), preferring higher FTP
        # and available headroom w.r.t. max pull.
        avgW2 = avg_power_for_pulls(P, pulls)
        frac2 = avgW2 / ftps
        slack = cap_fraction - frac2  # positive means headroom
        # Candidate receivers: those with slack>0 and headroom on pull duration
        candidates = []
        for j in range(n):
            if j == i_bad:
                continue
            max_allowed = strongest_pull_bounds[1] if j == 0 else other_pull_max_s
            if pulls[j] + 1e-9 < max_allowed and slack[j] > 0:
                # Score: prefer high slack and high FTP
                candidates.append((slack[j] * (ftps[j] / ftps.max()), j))

        if not candidates:
            # Nowhere to put time safely => infeasible
            return False, pulls, avgW2, frac2

        _, j_best = max(candidates, key=lambda x: x[0])
        pulls[j_best] += delta
        clamp(j_best)

    # Max iters reached: treat as infeasible
    avgW = avg_power_for_pulls(P, pulls)
    return bool(np.all(avgW <= cap_fraction * ftps + 1e-3)), pulls, avgW, avgW / ftps


def search_max_speed_plan(
    riders: List[Rider],
    crr: float,
    rho: float,
    draft_factors: List[float],
    cap_fraction: float,
    strongest_pull_bounds: Tuple[float, float],
    other_pull_max_s: float,
    min_pull_s: float,
    allow_zero_pull: bool,
    v_min_kph: float = 30.0,
    v_max_kph: float = 55.0,
    coarse_step_kph: float = 0.25,
    refine_iters: int = 14,
) -> Dict:
    """
    Coarse scan upward in speed, then binary-refine around the edge of feasibility.
    Returns dict with best plan found (even if none feasible, returns best attempt at v_min).
    """
    def kph_to_mps(kph: float) -> float:
        return kph / 3.6

    best = None

    # Coarse sweep
    last_feasible = None
    last_infeasible = None
    for kph in np.arange(v_min_kph, v_max_kph + 1e-9, coarse_step_kph):
        v = kph_to_mps(float(kph))
        feas, pulls, avgW, frac = try_find_feasible_pulls(
            riders=riders,
            v_mps=v,
            crr=crr,
            rho=rho,
            draft_factors=draft_factors,
            cap_fraction=cap_fraction,
            strongest_pull_bounds=strongest_pull_bounds,
            other_pull_max_s=other_pull_max_s,
            min_pull_s=min_pull_s,
            allow_zero_pull=allow_zero_pull,
        )
        if feas:
            last_feasible = (v, pulls, avgW, frac)
            best = last_feasible
        else:
            last_infeasible = (v, pulls, avgW, frac)
            if last_feasible is not None:
                break

    if best is None:
        # Nothing feasible in range; return lowest-speed attempt for diagnostics
        v = kph_to_mps(v_min_kph)
        feas, pulls, avgW, frac = try_find_feasible_pulls(
            riders=riders,
            v_mps=v,
            crr=crr,
            rho=rho,
            draft_factors=draft_factors,
            cap_fraction=cap_fraction,
            strongest_pull_bounds=strongest_pull_bounds,
            other_pull_max_s=other_pull_max_s,
            min_pull_s=min_pull_s,
            allow_zero_pull=allow_zero_pull,
        )
        return {
            "feasible": bool(feas),
            "v_mps": v,
            "v_kph": v_min_kph,
            "pulls_s": pulls,
            "avgW": avgW,
            "avgFrac": frac,
            "note": "No feasible plan found in the scanned speed range.",
        }

    # Refine with binary search between last feasible and first infeasible (if we have it)
    if last_infeasible is None:
        v, pulls, avgW, frac = best
        return {
            "feasible": True,
            "v_mps": v,
            "v_kph": 3.6 * v,
            "pulls_s": pulls,
            "avgW": avgW,
            "avgFrac": frac,
            "note": "Feasible up to the max scanned speed; increase v_max_kph to search further.",
        }

    v_lo, pulls_lo, avgW_lo, frac_lo = last_feasible
    v_hi, *_ = last_infeasible

    pulls_best, avgW_best, frac_best = pulls_lo, avgW_lo, frac_lo
    v_best = v_lo

    for _ in range(refine_iters):
        v_mid = 0.5 * (v_lo + v_hi)
        feas, pulls, avgW, frac = try_find_feasible_pulls(
            riders=riders,
            v_mps=v_mid,
            crr=crr,
            rho=rho,
            draft_factors=draft_factors,
            cap_fraction=cap_fraction,
            strongest_pull_bounds=strongest_pull_bounds,
            other_pull_max_s=other_pull_max_s,
            min_pull_s=min_pull_s,
            allow_zero_pull=allow_zero_pull,
        )
        if feas:
            v_lo = v_mid
            v_best = v_mid
            pulls_best, avgW_best, frac_best = pulls, avgW, frac
        else:
            v_hi = v_mid

    return {
        "feasible": True,
        "v_mps": v_best,
        "v_kph": 3.6 * v_best,
        "pulls_s": pulls_best,
        "avgW": avgW_best,
        "avgFrac": frac_best,
        "note": "Max-speed feasible plan (per the heuristic solver).",
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Zwift TTT Pull & Power Planner", layout="wide")
st.title("Zwift Indoor TTT Pull & Power Planner")

st.markdown(
    """
This app computes a constant-speed team time trial plan with:
- Fixed rider order: **strongest → weakest** (sorted by FTP by default)
- Drafting via **CdA reduction by position**
- Constraint: **each rider’s average power over a full rotation ≤ (cap × FTP)**
- Riders may **surge above FTP on the front**; only the rotation-average constraint is enforced.
"""
)

with st.sidebar:
    st.header("Environment / Model")
    crr = st.number_input("CRR", value=0.004, step=0.0005, format="%.4f")
    rho = st.number_input("Air density ρ (kg/m³)", value=1.214, step=0.01, format="%.3f")

    st.header("Constraints")
    cap_fraction = st.slider("Average cap (%FTP)", min_value=0.70, max_value=1.05, value=0.97, step=0.01)

    st.subheader("Pull time bounds")
    strongest_min = st.number_input("Strongest rider min pull (s)", value=60.0, step=5.0)
    strongest_max = st.number_input("Strongest rider max pull (s)", value=100.0, step=5.0)
    other_max = st.number_input("Other riders max pull (s)", value=120.0, step=5.0)
    min_pull = st.number_input("Min pull for non-zero pulls (s)", value=5.0, step=1.0)

    allow_zero = st.checkbox("Allow riders to skip the front (0 s pulls)", value=True)

    st.header("Speed search")
    vmin = st.number_input("Min speed to scan (km/h)", value=30.0, step=1.0)
    vmax = st.number_input("Max speed to scan (km/h)", value=55.0, step=1.0)
    step = st.number_input("Coarse step (km/h)", value=0.25, step=0.05, format="%.2f")

st.header("Riders")

n = st.number_input("Number of riders", min_value=4, max_value=8, value=4, step=1)

default_rows = []
for i in range(int(n)):
    default_rows.append(
        {
            "Name": f"R{i+1}",
            "Height_cm": 180.0 - 2.0 * i,
            "Weight_kg": 80.0 - 5.0 * i,
            "FTP_W": 300.0 - 20.0 * i,
            "Bike_kg": 8.0,
            "Cd": 0.69,
        }
    )
df_in = pd.DataFrame(default_rows)

st.caption("Enter rider stats. Rider order will be sorted strongest → weakest by FTP (ties by CdA).")
edited = st.data_editor(
    df_in,
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
)

# Build riders + compute CdA
riders: List[Rider] = []
for _, row in edited.iterrows():
    riders.append(
        Rider(
            name=str(row["Name"]),
            height_cm=float(row["Height_cm"]),
            weight_kg=float(row["Weight_kg"]),
            ftp_w=float(row["FTP_W"]),
            bike_kg=float(row.get("Bike_kg", 8.0)),
            cd=float(row.get("Cd", 0.69)),
        )
    )

# Sort strongest->weakest (FTP desc). If tie, lower CdA first (more aero) then higher weight (minor).
riders.sort(key=lambda r: (-r.ftp_w, r.cda_front, -r.weight_kg))

st.subheader("Sorted rider order (strongest → weakest)")
df_order = pd.DataFrame(
    [
        {
            "Order": i + 1,
            "Name": r.name,
            "FTP_W": r.ftp_w,
            "Weight_kg": r.weight_kg,
            "Height_cm": r.height_cm,
            "Bike_kg": r.bike_kg,
            "Cd": r.cd,
            "Area_m2": frontal_area_m2(r.weight_kg, r.height_cm),
            "CdA_m2": r.cda_front,
        }
        for i, r in enumerate(riders)
    ]
)
st.dataframe(df_order, use_container_width=True, hide_index=True)

st.subheader("Draft model")
draft = build_default_draft_factors(len(riders))
draft_df = pd.DataFrame(
    [{"Position": i + 1, "CdA factor": draft[i]} for i in range(len(draft))]
)
draft_df = st.data_editor(draft_df, use_container_width=True, hide_index=True)
draft_factors = [float(x) for x in draft_df["CdA factor"].tolist()]

if st.button("Compute max-speed plan"):
    plan = search_max_speed_plan(
        riders=riders,
        crr=float(crr),
        rho=float(rho),
        draft_factors=draft_factors,
        cap_fraction=float(cap_fraction),
        strongest_pull_bounds=(float(strongest_min), float(strongest_max)),
        other_pull_max_s=float(other_max),
        min_pull_s=float(min_pull),
        allow_zero_pull=bool(allow_zero),
        v_min_kph=float(vmin),
        v_max_kph=float(vmax),
        coarse_step_kph=float(step),
        refine_iters=16,
    )

    st.success(plan["note"])
    st.metric("Target speed (km/h)", f"{plan['v_kph']:.2f}")
    st.metric("Target speed (m/s)", f"{plan['v_mps']:.3f}")

    pulls = plan["pulls_s"]
    avgW = plan["avgW"]
    frac = plan["avgFrac"]

    # Detailed power by position at that speed
    P = compute_power_matrix(riders, plan["v_mps"], float(crr), float(rho), draft_factors)

    # Power required in each position for each rider (pos1..posN)
    # We'll compute pos-based directly for clarity:
    pos_powers = []
    for i, r in enumerate(riders):
        row = {"Name": r.name, "FTP_W": r.ftp_w}
        for pos in range(1, len(riders) + 1):
            cda_eff = r.cda_front * draft_factors[pos - 1]
            row[f"Pos{pos}_W"] = power_required_w(
                v_mps=plan["v_mps"],
                mass_kg=r.system_mass_kg,
                crr=float(crr),
                rho=float(rho),
                cda_eff=cda_eff,
            )
        pos_powers.append(row)
    df_pos = pd.DataFrame(pos_powers)

    df_plan = pd.DataFrame(
        [
            {
                "Order": i + 1,
                "Name": riders[i].name,
                "Pull_s": pulls[i],
                "Pull_share_%": 100.0 * pulls[i] / max(1e-9, pulls.sum()),
                "Avg_W_over_rotation": avgW[i],
                "Avg_%FTP": 100.0 * frac[i],
                "FTP_W": riders[i].ftp_w,
            }
            for i in range(len(riders))
        ]
    )

    st.subheader("Pull plan & rotation-average load")
    st.dataframe(df_plan, use_container_width=True, hide_index=True)

    st.subheader("Power required by position at target speed")
    st.dataframe(df_pos, use_container_width=True, hide_index=True)

    st.subheader("Rotation timing (one full cycle)")
    T = pulls.sum()
    timeline = []
    t0 = 0.0
    for k in range(len(riders)):
        leader = riders[k].name
        dt = pulls[k]
        if dt <= 1e-9:
            continue
        timeline.append(
            {
                "Segment": len(timeline) + 1,
                "Leader (Pos1)": leader,
                "Start_s": t0,
                "End_s": t0 + dt,
                "Duration_s": dt,
            }
        )
        t0 += dt
    df_time = pd.DataFrame(timeline)
    st.dataframe(df_time, use_container_width=True, hide_index=True)

    # If infeasible, show who breaks the cap
    if not plan["feasible"]:
        st.warning("Plan is infeasible under the current constraints. See Avg_%FTP above for the limiting rider(s).")
