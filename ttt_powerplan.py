# ttt_powerplan.py
# Streamlit app: Zwift TTT pull & power planner + local rider database (SQLite)
#
# Run:
#   pip install -r requirements.txt
#   streamlit run ttt_powerplan.py

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

G = 9.80665


# =============================
# Aero / physics
# =============================
def frontal_area_m2(weight_kg: float, height_cm: float) -> float:
    """
    User-provided polynomial for frontal area.
    IMPORTANT: height input is in cm.
    """
    W = float(weight_kg)
    H = float(height_cm)
    return (
        0.002112 * W
        - 0.00763 * H
        - 0.00000963 * (W**2)
        + 0.00001 * W * H
        + 0.0000233 * (H**2)
        + 0.7654
    )


def cda_m2(weight_kg: float, height_cm: float, cd: float) -> float:
    return float(cd) * frontal_area_m2(weight_kg, height_cm)


def power_required_w(v_mps: float, mass_kg: float, crr: float, rho: float, cda_eff: float) -> float:
    # Flat, no wind: P = v*(m g Crr) + 0.5*rho*CdA*v^3
    rolling = v_mps * (mass_kg * G * crr)
    aero = 0.5 * rho * cda_eff * (v_mps**3)
    return rolling + aero


def build_default_draft_factors(n_riders: int) -> List[float]:
    """
    Draft factors per position (CdA multiplier):
      pos1=1.00, pos2=0.75, pos3=0.70, pos4=0.67
    For n>4 we cap at 0.67 for deeper positions by default.
    """
    base = [1.00, 0.75, 0.70, 0.67]
    if n_riders <= 4:
        return base[:n_riders]
    return base + [0.67] * (n_riders - 4)


# =============================
# Rider / rotation model
# =============================
@dataclass
class Rider:
    name: str
    weight_kg: float
    height_cm: float
    ftp_w: float
    bike_kg: float = 8.0
    cd: float = 0.69

    @property
    def system_mass_kg(self) -> float:
        return float(self.weight_kg) + float(self.bike_kg)

    @property
    def cda_front(self) -> float:
        return cda_m2(self.weight_kg, self.height_cm, self.cd)


def position_of_rider_during_lead_segment(i: int, k: int, n: int) -> int:
    """
    Segment k: rider k leads (position 1). Then positions are cyclic:
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
    P[i,k] = power rider i must do during segment where rider k leads.
    """
    n = len(riders)
    P = np.zeros((n, n), dtype=float)
    for i, r in enumerate(riders):
        for k in range(n):
            pos = position_of_rider_during_lead_segment(i, k, n)
            cda_eff = r.cda_front * float(draft_factors[pos - 1])
            P[i, k] = power_required_w(v_mps, r.system_mass_kg, crr, rho, cda_eff)
    return P


def avg_power_for_pulls(P: np.ndarray, pulls_s: np.ndarray) -> np.ndarray:
    T = float(pulls_s.sum())
    if T <= 1e-12:
        return np.zeros(P.shape[0], dtype=float)
    return (P @ pulls_s) / T


# =============================
# Ordering rule: strongest -> weakest
# =============================
def solve_front_speed_mps(r: Rider, p_cap_w: float, crr: float, rho: float) -> float:
    """
    Solve v such that P_front(v) = p_cap_w with bisection.
    This is a physically grounded proxy for "who is strongest for setting pace on the flat",
    because it accounts for FTP, CdA and system mass.
    """
    m = r.system_mass_kg
    cda = r.cda_front
    lo, hi = 0.0, 30.0  # m/s upper bound is generous
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        p = mid * (m * G * crr) + 0.5 * rho * cda * (mid**3)
        if p < p_cap_w:
            lo = mid
        else:
            hi = mid
    return lo


def sort_riders_strong_to_weak(riders: List[Rider], cap_fraction: float, crr: float, rho: float) -> List[Rider]:
    # Higher sustainable front speed (at cap_fraction*FTP) => earlier in line
    return sorted(
        riders,
        key=lambda r: -solve_front_speed_mps(r, p_cap_w=float(cap_fraction) * float(r.ftp_w), crr=float(crr), rho=float(rho)),
    )


# =============================
# Solver: find max-speed feasible plan (heuristic)
# =============================
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
    max_iters: int = 2500,
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """
    Heuristic pull-time adjustment to satisfy avg constraint:
      avg_i = sum_k t_k*P[i,k] / sum_k t_k  <= cap_fraction * FTP_i
    Strongest rider (index 0) is constrained to strongest_pull_bounds.
    Others to [0|min_pull_s, other_pull_max_s], depending on allow_zero_pull.
    """
    n = len(riders)
    if n < 2:
        return False, np.zeros(n), np.zeros(n), np.zeros(n)

    P = compute_power_matrix(riders, v_mps, crr, rho, draft_factors)
    ftps = np.array([r.ftp_w for r in riders], dtype=float)
    caps = cap_fraction * ftps

    tmin = 0.0 if allow_zero_pull else float(min_pull_s)
    pulls = np.zeros(n, dtype=float)

    # Initial guess: strongest at midpoint; others ramp by FTP but allow weakest to be low/zero.
    pulls[0] = 0.5 * (float(strongest_pull_bounds[0]) + float(strongest_pull_bounds[1]))
    ftp_norm = (ftps - ftps.min()) / max(1e-9, (ftps.max() - ftps.min()))
    for i in range(1, n):
        base = 20.0 + 50.0 * ftp_norm[i]  # 20..70s
        pulls[i] = max(tmin, min(base, float(other_pull_max_s)))
    if allow_zero_pull:
        pulls[-1] = 0.0

    def clamp(i: int) -> None:
        if i == 0:
            pulls[i] = float(np.clip(pulls[i], strongest_pull_bounds[0], strongest_pull_bounds[1]))
        else:
            pulls[i] = float(np.clip(pulls[i], tmin, other_pull_max_s))

    for i in range(n):
        clamp(i)

    if pulls.sum() <= 1e-9:
        return False, pulls, np.zeros(n), np.zeros(n)

    for _ in range(max_iters):
        avgW = avg_power_for_pulls(P, pulls)
        viol = avgW - caps
        if np.all(viol <= 1e-6):
            return True, pulls, avgW, avgW / ftps

        i_bad = int(np.argmax(viol))
        min_allowed = strongest_pull_bounds[0] if i_bad == 0 else tmin
        if pulls[i_bad] <= min_allowed + 1e-9:
            return False, pulls, avgW, avgW / ftps

        delta = min(2.0, pulls[i_bad] - min_allowed)  # seconds
        pulls[i_bad] -= delta
        clamp(i_bad)

        # Recompute slack, then assign delta to best receiver.
        avgW2 = avg_power_for_pulls(P, pulls)
        frac2 = avgW2 / ftps
        slack = cap_fraction - frac2

        candidates = []
        for j in range(n):
            if j == i_bad:
                continue
            max_allowed = strongest_pull_bounds[1] if j == 0 else other_pull_max_s
            if pulls[j] + 1e-9 < max_allowed and slack[j] > 0:
                # Prefer high slack and high FTP
                candidates.append((slack[j] * (ftps[j] / ftps.max()), j))

        if not candidates:
            return False, pulls, avgW2, frac2

        _, j_best = max(candidates, key=lambda x: x[0])
        pulls[j_best] += delta
        clamp(j_best)

    avgW = avg_power_for_pulls(P, pulls)
    return bool(np.all(avgW <= caps + 1e-3)), pulls, avgW, avgW / ftps


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
    v_max_kph: float = 60.0,
    coarse_step_kph: float = 0.25,
    refine_iters: int = 16,
) -> Dict:
    def kph_to_mps(kph: float) -> float:
        return kph / 3.6

    best = None
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
            "v_kph": float(v_min_kph),
            "pulls_s": pulls,
            "avgW": avgW,
            "avgFrac": frac,
            "note": "No feasible plan found in the scanned speed range.",
        }

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

    v_best, pulls_best, avgW_best, frac_best = v_lo, pulls_lo, avgW_lo, frac_lo

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
            v_best, pulls_best, avgW_best, frac_best = v_mid, pulls, avgW, frac
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


# =============================
# Results table helpers
# =============================
def build_combined_results_table(riders: List[Rider], pulls: np.ndarray, P: np.ndarray, avgW: np.ndarray) -> pd.DataFrame:
    """
    Combined table (starting order):
      - Pull time
      - Pull (front) power
      - Avg drafting power (time-weighted over non-front segments)
      - Overall avg power
      - %FTP
    Displayed as whole numbers.
    """
    n = len(riders)
    T = float(pulls.sum())
    rows = []
    for i, r in enumerate(riders):
        t_front = float(pulls[i])
        p_front = float(P[i, i])  # during segment where rider i leads

        t_draft = max(0.0, T - t_front)
        if t_draft > 1e-9:
            draft_work = float(np.dot(pulls, P[i, :]) - t_front * p_front)
            p_draft_avg = draft_work / t_draft
        else:
            p_draft_avg = 0.0

        rows.append(
            {
                "Order": i + 1,
                "Rider": r.name,
                "Pull_s": int(round(t_front)),
                "Pull_W": int(round(p_front)),
                "DraftAvg_W": int(round(p_draft_avg)),
                "Avg_W": int(round(float(avgW[i]))),
                "%FTP": int(round(100.0 * float(avgW[i]) / float(r.ftp_w))),
            }
        )
    return pd.DataFrame(rows)


def build_position_power_table(riders: List[Rider], v_mps: float, crr: float, rho: float, draft_factors: List[float]) -> pd.DataFrame:
    n = len(riders)
    rows = []
    for r in riders:
        row = {"Rider": r.name, "FTP_W": int(round(r.ftp_w))}
        for pos in range(1, n + 1):
            cda_eff = r.cda_front * float(draft_factors[pos - 1])
            row[f"Pos{pos}_W"] = int(round(power_required_w(v_mps, r.system_mass_kg, crr, rho, cda_eff)))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================
# Local SQLite rider database
# =============================
def db_path() -> Path:
    # Local to the script directory for portability.
    return Path(__file__).with_name("riders.sqlite3")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS riders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                height_cm REAL NOT NULL,
                weight_kg REAL NOT NULL,
                ftp_w REAL NOT NULL,
                bike_kg REAL NOT NULL DEFAULT 8,
                cd REAL NOT NULL DEFAULT 0.69
            )
            """
        )
        conn.commit()


def fetch_riders_df() -> pd.DataFrame:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT id, name, height_cm, weight_kg, ftp_w, bike_kg, cd FROM riders ORDER BY name ASC").fetchall()
    if not rows:
        return pd.DataFrame(columns=["id", "name", "height_cm", "weight_kg", "ftp_w", "bike_kg", "cd"])
    return pd.DataFrame([dict(r) for r in rows])


def upsert_rider(row: Dict) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO riders (name, height_cm, weight_kg, ftp_w, bike_kg, cd)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                height_cm=excluded.height_cm,
                weight_kg=excluded.weight_kg,
                ftp_w=excluded.ftp_w,
                bike_kg=excluded.bike_kg,
                cd=excluded.cd
            """,
            (row["name"], row["height_cm"], row["weight_kg"], row["ftp_w"], row["bike_kg"], row["cd"]),
        )
        conn.commit()


def delete_rider_by_name(name: str) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM riders WHERE name = ?", (name,))
        conn.commit()


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Zwift TTT Power Planner", layout="wide")
st.title("Zwift TTT Power Planner")

tabs = st.tabs(["Power Plan", "Rider Database"])

# -----------------------------
# Rider database tab
# -----------------------------
with tabs[1]:
    st.subheader("Local rider database")
    st.caption("Stored locally in SQLite next to this script. Edit values and click **Save**. Tick Delete to remove a rider.")

    db_df = fetch_riders_df()

    if db_df.empty:
        st.info("Database is empty. Add riders below.")
        db_df = pd.DataFrame(columns=["id", "name", "height_cm", "weight_kg", "ftp_w", "bike_kg", "cd"])

    # Editable grid
    edit_df = db_df.copy()
    if "delete" not in edit_df.columns:
        edit_df.insert(0, "delete", False)

    edited_db = st.data_editor(
        edit_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "id": st.column_config.NumberColumn("id", disabled=True),
            "delete": st.column_config.CheckboxColumn("Delete"),
            "name": st.column_config.TextColumn("name"),
            "height_cm": st.column_config.NumberColumn("height_cm"),
            "weight_kg": st.column_config.NumberColumn("weight_kg"),
            "ftp_w": st.column_config.NumberColumn("ftp_w"),
            "bike_kg": st.column_config.NumberColumn("bike_kg"),
            "cd": st.column_config.NumberColumn("cd"),
        },
        key="db_editor",
    )

    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("Save changes to DB"):
            # Apply deletes first
            for _, r in edited_db.iterrows():
                name = str(r.get("name", "")).strip()
                if not name:
                    continue
                if bool(r.get("delete", False)):
                    delete_rider_by_name(name)

            # Upsert all non-deleted rows with valid data
            for _, r in edited_db.iterrows():
                if bool(r.get("delete", False)):
                    continue
                name = str(r.get("name", "")).strip()
                if not name:
                    continue
                row = {
                    "name": name,
                    "height_cm": float(r["height_cm"]),
                    "weight_kg": float(r["weight_kg"]),
                    "ftp_w": float(r["ftp_w"]),
                    "bike_kg": float(r.get("bike_kg", 8.0)),
                    "cd": float(r.get("cd", 0.69)),
                }
                upsert_rider(row)

            st.success("Database updated.")
            st.rerun()

    with colB:
        st.caption("Tip: keep rider names unique. The app uses names for selection & updating.")

# -----------------------------
# Power plan tab
# -----------------------------
with tabs[0]:
    with st.sidebar:
        st.header("Environment")
        crr = st.number_input("CRR", value=0.004, step=0.0005, format="%.4f")
        rho = st.number_input("Air density Ï (kg/mÂ³)", value=1.214, step=0.01, format="%.3f")

        st.header("Constraints")
        cap_fraction = st.slider("Rotation-average cap (%FTP)", 0.70, 1.05, 0.97, 0.01)

        st.subheader("Pull bounds (solver)")
        strongest_min = st.number_input("Strongest min pull (s)", value=60.0, step=5.0)
        strongest_max = st.number_input("Strongest max pull (s)", value=100.0, step=5.0)
        other_max = st.number_input("Other riders max pull (s)", value=120.0, step=5.0)
        min_pull = st.number_input("Min pull for non-zero pulls (s)", value=5.0, step=1.0)
        allow_zero = st.checkbox("Allow riders to skip the front (0 s pulls)", value=True)

        st.header("Speed search")
        vmin = st.number_input("Min speed scan (km/h)", value=30.0, step=1.0)
        vmax = st.number_input("Max speed scan (km/h)", value=60.0, step=1.0)
        step = st.number_input("Coarse step (km/h)", value=0.25, step=0.05, format="%.2f")

    st.subheader("Select riders for this plan")

    all_db = fetch_riders_df()
    if all_db.empty:
        st.warning("Your rider database is empty. Add riders in the Rider Database tab.")
        st.stop()

    names = all_db["name"].tolist()
    selected_names = st.multiselect("Pick riders (4â8)", options=names, default=names[:4])

    if len(selected_names) < 4:
        st.info("Select at least 4 riders to compute a plan.")
        st.stop()
    if len(selected_names) > 8:
        st.info("Select no more than 8 riders.")
        st.stop()

    sel_df = all_db[all_db["name"].isin(selected_names)].copy()
    sel_df = sel_df.sort_values("name").reset_index(drop=True)

    st.caption("You can temporarily tweak rider values here before solving (changes are NOT saved to the DB).")
    tmp_edit = st.data_editor(
        sel_df.drop(columns=["id"]),
        use_container_width=True,
        hide_index=True,
        key="tmp_riders_editor",
    )

    # Build Rider objects from edited values
    riders_raw: List[Rider] = []
    for _, r in tmp_edit.iterrows():
        riders_raw.append(
            Rider(
                name=str(r["name"]),
                height_cm=float(r["height_cm"]),
                weight_kg=float(r["weight_kg"]),
                ftp_w=float(r["ftp_w"]),
                bike_kg=float(r.get("bike_kg", 8.0)),
                cd=float(r.get("cd", 0.69)),
            )
        )

    # Order strongest->weakest using FTP + CdA + mass proxy (front-speed at cap fraction)
    riders = sort_riders_strong_to_weak(riders_raw, cap_fraction=cap_fraction, crr=crr, rho=rho)

    st.subheader("Starting order (strongest â weakest)")
    order_df = pd.DataFrame(
        [
            {
                "Order": i + 1,
                "Rider": r.name,
                "FTP_W": int(round(r.ftp_w)),
                "Weight_kg": int(round(r.weight_kg)),
                "Height_cm": int(round(r.height_cm)),
                "Bike_kg": int(round(r.bike_kg)),
                "Cd": r.cd,
                "Area_x1000": int(round(1000.0 * frontal_area_m2(r.weight_kg, r.height_cm))),  # shown as integer
                "CdA_x1000": int(round(1000.0 * r.cda_front)),
            }
            for i, r in enumerate(riders)
        ]
    )
    st.dataframe(order_df, use_container_width=True, hide_index=True)

    st.subheader("Draft factors by position")
    draft_default = build_default_draft_factors(len(riders))
    draft_df = pd.DataFrame([{"Position": i + 1, "CdA_factor": float(draft_default[i])} for i in range(len(riders))])
    draft_df = st.data_editor(draft_df, use_container_width=True, hide_index=True, key="draft_editor")
    draft_factors = [float(x) for x in draft_df["CdA_factor"].tolist()]

    if st.button("Compute max-speed plan", type="primary"):
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
            refine_iters=18,
        )
        st.session_state["plan"] = plan
        st.session_state["riders"] = riders
        st.session_state["draft_factors"] = draft_factors
        st.session_state["env"] = {"crr": float(crr), "rho": float(rho), "cap_fraction": float(cap_fraction)}

    if "plan" not in st.session_state:
        st.stop()

    plan = st.session_state["plan"]
    riders = st.session_state["riders"]
    draft_factors = st.session_state["draft_factors"]
    env = st.session_state["env"]
    crr = env["crr"]
    rho = env["rho"]
    cap_fraction = env["cap_fraction"]

    st.success(plan["note"])
    st.metric("Target speed (km/h)", int(round(plan["v_kph"])))
    st.metric("Target speed (m/s)", int(round(plan["v_mps"])))

    # Compute matrices at target speed
    pulls = plan["pulls_s"].copy()
    P = compute_power_matrix(riders, plan["v_mps"], crr, rho, draft_factors)
    avgW = avg_power_for_pulls(P, pulls)

    st.subheader("Combined rider plan (editable pull times)")
    df_combined = build_combined_results_table(riders, pulls, P, avgW)

    st.caption("Edit Pull_s to adjust durations manually, then click Recalculate.")
    df_edit = st.data_editor(
        df_combined,
        use_container_width=True,
        hide_index=True,
        key="combined_editor",
        column_config={
            "Pull_s": st.column_config.NumberColumn("Pull_s", min_value=0, step=1),
        },
        disabled=["Order", "Rider", "Pull_W", "DraftAvg_W", "Avg_W", "%FTP"],
    )

    # Streamlit's data_editor doesn't easily allow "only one column editable and compute rest" cleanly,
    # so we take Pull_s from df_edit and recompute everything when user clicks.
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Recalculate with manual pulls"):
            new_pulls = np.array([float(x) for x in df_edit["Pull_s"].tolist()], dtype=float)
            new_pulls = np.clip(new_pulls, 0.0, None)
            if new_pulls.sum() <= 1e-9:
                st.error("Total pull time is zero. Set at least one rider to a non-zero pull.")
            else:
                avgW2 = avg_power_for_pulls(P, new_pulls)
                df_new = build_combined_results_table(riders, new_pulls, P, avgW2)
                st.session_state["manual_pulls"] = new_pulls
                st.session_state["combined_table"] = df_new
                st.rerun()

    with col2:
        st.caption("Manual pulls keep the same target speed; the table updates %FTP and drafting averages accordingly.")

    # Display recalculated combined table if present
    if "combined_table" in st.session_state and "manual_pulls" in st.session_state:
        df_out = st.session_state["combined_table"]
        pulls_used = st.session_state["manual_pulls"]
        st.dataframe(df_out, use_container_width=True, hide_index=True)
    else:
        df_out = df_combined
        pulls_used = pulls

    # Additional tables (whole numbers)
    st.subheader("Power required by position (whole numbers)")
    df_pos = build_position_power_table(riders, plan["v_mps"], crr, rho, draft_factors)
    st.dataframe(df_pos, use_container_width=True, hide_index=True)

    st.subheader("Rotation timeline (one cycle)")
    t0 = 0.0
    timeline = []
    for k, r in enumerate(riders):
        dt = float(pulls_used[k])
        if dt <= 1e-9:
            continue
        timeline.append(
            {
                "Segment": len(timeline) + 1,
                "Leader (Pos1)": r.name,
                "Start_s": int(round(t0)),
                "End_s": int(round(t0 + dt)),
                "Duration_s": int(round(dt)),
            }
        )
        t0 += dt
    st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)

    # Constraint check after manual edits
    st.subheader("Constraint check")
    ftps = np.array([r.ftp_w for r in riders], dtype=float)
    avgW_used = avg_power_for_pulls(P, pulls_used)
    frac_used = avgW_used / ftps
    chk = pd.DataFrame(
        [
            {
                "Order": i + 1,
                "Rider": riders[i].name,
                "Avg_W": int(round(avgW_used[i])),
                "%FTP": int(round(100.0 * frac_used[i])),
                "Cap_%FTP": int(round(100.0 * cap_fraction)),
                "Over_cap": bool(avgW_used[i] > cap_fraction * ftps[i] + 1e-6),
            }
            for i in range(len(riders))
        ]
    )
    st.dataframe(chk, use_container_width=True, hide_index=True)
