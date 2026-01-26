# ttt_powerplan.py
# Streamlit app: Zwift TTT pull & power planner + local rider database (SQLite)
#
# Run:
#   pip install -r requirements.txt
#   streamlit run ttt_powerplan.py

from __future__ import annotations

import sqlite3
import io
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

# =============================
# Local database (SQLite)
# =============================
def db_path() -> Path:
    # Local to the script directory for portability.
    return Path(__file__).with_name("ttt_riders.sqlite3")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Basic safety
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def init_db() -> None:
    """Create tables and migrate from older single-table schema if detected."""
    with get_conn() as conn:
        # Bikes table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bikes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                bike_kg REAL NOT NULL DEFAULT 8,
                cd REAL NOT NULL DEFAULT 0.69
            )
            """
        )

        # Detect if an older riders table exists and contains bike_kg/cd directly
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS riders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                height_cm REAL NOT NULL,
                weight_kg REAL NOT NULL,
                ftp_w REAL NOT NULL,
                default_bike_id INTEGER,
                FOREIGN KEY(default_bike_id) REFERENCES bikes(id) ON DELETE SET NULL
            )
            """
        )

        # Migration: if riders table has legacy columns bike_kg and cd, migrate to bikes table.
        cols = _table_columns(conn, "riders")
        if ("bike_kg" in cols) or ("cd" in cols):
            # Create a new riders table with correct schema
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS riders_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    height_cm REAL NOT NULL,
                    weight_kg REAL NOT NULL,
                    ftp_w REAL NOT NULL,
                    default_bike_id INTEGER,
                    FOREIGN KEY(default_bike_id) REFERENCES bikes(id) ON DELETE SET NULL
                )
                """
            )
            # Collect unique bikes from old rider rows
            old_rows = conn.execute(
                "SELECT name, height_cm, weight_kg, ftp_w, bike_kg, cd FROM riders"
            ).fetchall()

            bike_map: Dict[Tuple[float, float], int] = {}
            for r in old_rows:
                bk = float(r["bike_kg"]) if r["bike_kg"] is not None else 8.0
                cdv = float(r["cd"]) if r["cd"] is not None else 0.69
                key = (round(bk, 3), round(cdv, 4))
                if key not in bike_map:
                    # Ensure unique bike name
                    base = f"Imported {key[0]:g}kg Cd {key[1]:g}"
                    name = base
                    suffix = 2
                    while True:
                        try:
                            conn.execute(
                                "INSERT INTO bikes(name, bike_kg, cd) VALUES (?,?,?)",
                                (name, key[0], key[1]),
                            )
                            bike_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
                            bike_map[key] = bike_id
                            break
                        except sqlite3.IntegrityError:
                            name = f"{base} ({suffix})"
                            suffix += 1

            # Insert riders into new table
            for r in old_rows:
                key = (round(float(r["bike_kg"]), 3), round(float(r["cd"]), 4))
                bike_id = bike_map.get(key)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO riders_new(name, height_cm, weight_kg, ftp_w, default_bike_id)
                    VALUES (?,?,?,?,?)
                    """,
                    (r["name"], float(r["height_cm"]), float(r["weight_kg"]), float(r["ftp_w"]), bike_id),
                )

            # Replace old table
            conn.execute("DROP TABLE riders")
            conn.execute("ALTER TABLE riders_new RENAME TO riders")

        # Ensure at least one bike exists
        n_bikes = conn.execute("SELECT COUNT(*) AS n FROM bikes").fetchone()["n"]
        if n_bikes == 0:
            conn.execute("INSERT INTO bikes(name, bike_kg, cd) VALUES (?,?,?)", ("Default", 8.0, 0.69))

        conn.commit()


def fetch_bikes_df() -> pd.DataFrame:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, bike_kg, cd FROM bikes ORDER BY name ASC"
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["id", "name", "bike_kg", "cd"])
    return pd.DataFrame([dict(r) for r in rows])


def upsert_bike(name: str, bike_kg: float, cd: float) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO bikes(name, bike_kg, cd)
            VALUES (?,?,?)
            ON CONFLICT(name) DO UPDATE SET
              bike_kg=excluded.bike_kg,
              cd=excluded.cd
            """,
            (name.strip(), float(bike_kg), float(cd)),
        )
        conn.commit()


def delete_bike_by_name(name: str) -> None:
    init_db()
    with get_conn() as conn:
        n_bikes = conn.execute("SELECT COUNT(*) AS n FROM bikes").fetchone()["n"]
        if n_bikes <= 1:
            # Keep at least one bike
            return
        conn.execute("DELETE FROM bikes WHERE name=?", (name.strip(),))
        conn.commit()


def fetch_riders_df() -> pd.DataFrame:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
              r.id, r.name, r.height_cm, r.weight_kg, r.ftp_w,
              r.default_bike_id,
              b.name AS default_bike_name, b.bike_kg AS default_bike_kg, b.cd AS default_bike_cd
            FROM riders r
            LEFT JOIN bikes b ON b.id = r.default_bike_id
            ORDER BY r.name ASC
            """
        ).fetchall()
    if not rows:
        return pd.DataFrame(
            columns=[
                "id","name","height_cm","weight_kg","ftp_w",
                "default_bike_id","default_bike_name","default_bike_kg","default_bike_cd"
            ]
        )
    return pd.DataFrame([dict(r) for r in rows])


def upsert_rider(
    name: str,
    height_cm: float,
    weight_kg: float,
    ftp_w: float,
    default_bike_id: int | None,
) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO riders(name, height_cm, weight_kg, ftp_w, default_bike_id)
            VALUES (?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
              height_cm=excluded.height_cm,
              weight_kg=excluded.weight_kg,
              ftp_w=excluded.ftp_w,
              default_bike_id=excluded.default_bike_id
            """,
            (name.strip(), float(height_cm), float(weight_kg), float(ftp_w), default_bike_id),
        )
        conn.commit()


def delete_rider_by_name(name: str) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM riders WHERE name=?", (name.strip(),))
        conn.commit()


def export_bikes_csv() -> str:
    df = fetch_bikes_df()
    cols = ["name", "bike_kg", "cd"]
    return df[cols].to_csv(index=False)


def export_riders_csv() -> str:
    df = fetch_riders_df()
    cols = ["name", "height_cm", "weight_kg", "ftp_w", "default_bike_name"]
    if df.empty:
        return pd.DataFrame(columns=cols).to_csv(index=False)
    # Ensure column exists even if NULLs
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].to_csv(index=False)


def import_bikes_csv(csv_bytes: bytes) -> Tuple[int, List[str]]:
    """Import bikes from CSV with columns: name,bike_kg,cd"""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    required = {"name", "bike_kg", "cd"}
    if not required.issubset(set(df.columns.str.lower())):
        # try case-insensitive map
        df.columns = [c.lower() for c in df.columns]
    if not required.issubset(set(df.columns)):
        return 0, [f"CSV must contain columns: {sorted(required)}"]
    errors = []
    n_ok = 0
    for _, row in df.iterrows():
        try:
            upsert_bike(str(row["name"]), float(row["bike_kg"]), float(row["cd"]))
            n_ok += 1
        except Exception as e:
            errors.append(f"{row.get('name','(unknown)')}: {e}")
    return n_ok, errors


def import_riders_csv(csv_bytes: bytes) -> Tuple[int, List[str]]:
    """Import riders from CSV with columns: name,height_cm,weight_kg,ftp_w,default_bike_name (optional)."""
    init_db()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = [c.lower() for c in df.columns]
    required = {"name", "height_cm", "weight_kg", "ftp_w"}
    if not required.issubset(set(df.columns)):
        return 0, [f"CSV must contain columns: {sorted(required)}"]
    bikes = fetch_bikes_df()
    bike_name_to_id = {str(r["name"]): int(r["id"]) for _, r in bikes.iterrows()}
    default_bike_id = int(bikes.iloc[0]["id"]) if len(bikes) else None

    errors = []
    n_ok = 0
    for _, row in df.iterrows():
        try:
            bike_id = None
            if "default_bike_name" in df.columns and pd.notna(row.get("default_bike_name")):
                bn = str(row.get("default_bike_name"))
                bike_id = bike_name_to_id.get(bn)
                if bike_id is None and bn.strip():
                    # Create a new bike entry with defaults if unknown
                    upsert_bike(bn.strip(), 8.0, 0.69)
                    bikes2 = fetch_bikes_df()
                    bike_name_to_id = {str(r["name"]): int(r["id"]) for _, r in bikes2.iterrows()}
                    bike_id = bike_name_to_id.get(bn.strip())
            if bike_id is None:
                bike_id = default_bike_id

            upsert_rider(
                name=str(row["name"]),
                height_cm=float(row["height_cm"]),
                weight_kg=float(row["weight_kg"]),
                ftp_w=float(row["ftp_w"]),
                default_bike_id=bike_id,
            )
            n_ok += 1
        except Exception as e:
            errors.append(f"{row.get('name','(unknown)')}: {e}")
    return n_ok, errors


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Zwift TTT Power Plan", layout="wide")
st.title("Zwift Indoor Team Time Trial Pull & Power Planner")

init_db()
df_riders_all = fetch_riders_df()
df_bikes_all = fetch_bikes_df()

tabs = st.tabs(["Power Plan", "Rider Database", "Bike Database"])

# -----------------------------
# Bike Database tab
# -----------------------------
with tabs[2]:
    st.header("Bike Database (local SQLite)")
    st.caption("Maintain bikes here. Rider records reference a default bike, but you can override bike choice per plan run.")

    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Add / Update bike")
        bike_name = st.text_input("Bike name", value="")
        bike_kg = st.number_input("Bike weight (kg)", value=8.0, step=0.1)
        bike_cd = st.number_input("Bike+rider Cd", value=0.69, step=0.01, format="%.3f")
        if st.button("Save bike", key="save_bike"):
            if bike_name.strip():
                upsert_bike(bike_name.strip(), bike_kg, bike_cd)
                st.success("Bike saved.")
                st.rerun()
            else:
                st.warning("Bike name cannot be empty.")

    with colB:
        st.subheader("Delete bike")
        bikes_df = fetch_bikes_df()
        if len(bikes_df) <= 1:
            st.info("At least one bike must exist; deletion disabled.")
        else:
            del_bike = st.selectbox("Select bike to delete", bikes_df["name"].tolist(), key="del_bike")
            if st.button("Delete selected bike", key="delete_bike"):
                delete_bike_by_name(del_bike)
                st.success("Bike deleted.")
                st.rerun()

    st.subheader("Current bikes")
    st.dataframe(fetch_bikes_df().assign(
        bike_kg=lambda d: d["bike_kg"].round(3),
        cd=lambda d: d["cd"].round(4),
    ), use_container_width=True, hide_index=True)

    st.subheader("Import / Export bikes (CSV)")
    exp = export_bikes_csv()
    st.download_button("Download bikes.csv", data=exp, file_name="bikes.csv", mime="text/csv")

    up = st.file_uploader("Import bikes.csv", type=["csv"], key="import_bikes")
    if up is not None:
        n_ok, errors = import_bikes_csv(up.read())
        if errors:
            st.warning("Imported with some issues:")
            for e in errors[:10]:
                st.write(f"- {e}")
            if len(errors) > 10:
                st.write(f"...and {len(errors)-10} more.")
        st.success(f"Imported/updated {n_ok} bikes.")
        st.rerun()

# -----------------------------
# Rider Database tab
# -----------------------------
with tabs[1]:
    st.header("Rider Database (local SQLite)")
    st.caption("Maintain rider anthropometrics and FTP here. Each rider can have a default bike (optional).")

    bikes_df = fetch_bikes_df()
    bike_name_to_id = {str(r["name"]): int(r["id"]) for _, r in bikes_df.iterrows()}
    bike_names = bikes_df["name"].tolist()
    default_bike_name = bike_names[0] if bike_names else "Default"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Add / Update rider")
        r_name = st.text_input("Name", value="")
        r_height = st.number_input("Height (cm)", value=180.0, step=1.0)
        r_weight = st.number_input("Weight (kg)", value=75.0, step=0.5)
        r_ftp = st.number_input("FTP (W)", value=280.0, step=5.0)
        r_bike_name = st.selectbox("Default bike", bike_names, index=0, key="r_default_bike") if bike_names else None

        if st.button("Save rider", key="save_rider"):
            if r_name.strip():
                bike_id = bike_name_to_id.get(r_bike_name) if r_bike_name else None
                upsert_rider(r_name.strip(), r_height, r_weight, r_ftp, bike_id)
                st.success("Rider saved.")
                st.rerun()
            else:
                st.warning("Rider name cannot be empty.")

    with col2:
        st.subheader("Delete rider")
        riders_df = fetch_riders_df()
        if riders_df.empty:
            st.info("No riders in database yet.")
        else:
            del_name = st.selectbox("Select rider to delete", riders_df["name"].tolist(), key="del_rider")
            if st.button("Delete selected rider", key="delete_rider"):
                delete_rider_by_name(del_name)
                st.success("Rider deleted.")
                st.rerun()

    st.subheader("Current riders")
    riders_df = fetch_riders_df()
    if not riders_df.empty:
        show = riders_df.copy()
        show["height_cm"] = show["height_cm"].round(0).astype(int)
        show["weight_kg"] = show["weight_kg"].round(1)
        show["ftp_w"] = show["ftp_w"].round(0).astype(int)
        show["default_bike_name"] = show["default_bike_name"].fillna(default_bike_name)
        st.dataframe(show[["name","height_cm","weight_kg","ftp_w","default_bike_name"]], use_container_width=True, hide_index=True)

    st.subheader("Import / Export riders (CSV)")
    exp = export_riders_csv()
    st.download_button("Download riders.csv", data=exp, file_name="riders.csv", mime="text/csv")

    up = st.file_uploader("Import riders.csv", type=["csv"], key="import_riders")
    if up is not None:
        n_ok, errors = import_riders_csv(up.read())
        if errors:
            st.warning("Imported with some issues:")
            for e in errors[:10]:
                st.write(f"- {e}")
            if len(errors) > 10:
                st.write(f"...and {len(errors)-10} more.")
        st.success(f"Imported/updated {n_ok} riders.")
        st.rerun()

# -----------------------------
# Power Plan tab
# -----------------------------
with tabs[0]:
    st.header("Power Plan")

    with st.sidebar:
        st.subheader("Environment / Model")
        crr = st.number_input("CRR", value=0.004, step=0.0005, format="%.4f")
        rho = st.number_input("Air density Ï (kg/mÂ³)", value=1.214, step=0.01, format="%.3f")

        st.subheader("Constraints")
        cap_fraction = st.slider("Average cap (%FTP)", min_value=0.70, max_value=1.05, value=0.97, step=0.01)

        st.subheader("Pull time bounds (solver)")
        strongest_min = st.number_input("Strongest min pull (s)", value=60.0, step=5.0)
        strongest_max = st.number_input("Strongest max pull (s)", value=100.0, step=5.0)
        other_max = st.number_input("Other riders max pull (s)", value=120.0, step=5.0)
        min_pull = st.number_input("Min pull for non-zero pulls (s)", value=5.0, step=1.0)
        allow_zero = st.checkbox("Allow riders to skip the front (0 s pulls)", value=True)

        st.subheader("Speed search")
        vmin = st.number_input("Min speed to scan (km/h)", value=30.0, step=1.0)
        vmax = st.number_input("Max speed to scan (km/h)", value=55.0, step=1.0)
        step = st.number_input("Coarse step (km/h)", value=0.25, step=0.05, format="%.2f")

    riders_df = fetch_riders_df()
    bikes_df = fetch_bikes_df()

    if riders_df.empty:
        st.warning("No riders found in the database. Add riders in the Rider Database tab first.")
        st.stop()
    if bikes_df.empty:
        st.warning("No bikes found in the database. Add bikes in the Bike Database tab first.")
        st.stop()

    bike_names = bikes_df["name"].tolist()
    bike_name_to_row = {str(r["name"]): r for _, r in bikes_df.iterrows()}
    bike_name_to_id = {str(r["name"]): int(r["id"]) for _, r in bikes_df.iterrows()}

    st.subheader("Select riders for this plan")
    selected_names = st.multiselect(
        "Pick riders (4â8). You can edit values temporarily below before solving.",
        options=riders_df["name"].tolist(),
        default=riders_df["name"].tolist()[:4],
    )
    if len(selected_names) < 4:
        st.info("Select at least 4 riders.")
        st.stop()
    if len(selected_names) > 8:
        st.info("Max 8 riders.")
        st.stop()

    # Build selection dataframe with defaults (including default bike)
    sel = riders_df[riders_df["name"].isin(selected_names)].copy()
    sel["Bike"] = sel["default_bike_name"].fillna(bike_names[0])
    sel.loc[~sel["Bike"].isin(bike_names), "Bike"] = bike_names[0]

    # Populate bike params from selected bike (but keep editable)
    sel["Bike_kg"] = sel["Bike"].map(lambda bn: float(bike_name_to_row[bn]["bike_kg"]))
    sel["Cd"] = sel["Bike"].map(lambda bn: float(bike_name_to_row[bn]["cd"]))

    # Editable: allow temporary changes to height/weight/ftp and bike choice/params
    st.subheader("Edit selected rider values (temporary overrides)")
    sel_edit = st.data_editor(
        sel[["name","height_cm","weight_kg","ftp_w","Bike","Bike_kg","Cd"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Rider", disabled=True),
            "height_cm": st.column_config.NumberColumn("Height (cm)", step=1, format="%.0f"),
            "weight_kg": st.column_config.NumberColumn("Weight (kg)", step=0.1, format="%.1f"),
            "ftp_w": st.column_config.NumberColumn("FTP (W)", step=1, format="%.0f"),
            "Bike": st.column_config.SelectboxColumn("Bike", options=bike_names),
            "Bike_kg": st.column_config.NumberColumn("Bike kg (override)", step=0.1, format="%.1f"),
            "Cd": st.column_config.NumberColumn("Cd (override)", step=0.001, format="%.3f"),
        },
        key="sel_edit",
    )

    # Order rule: strongest -> weakest using estimated cap-speed proxy (includes FTP, CdA, mass)
    def _solve_front_speed_mps(row) -> float:
        # Use cap_fraction * FTP, and current crr/rho for ordering.
        ftp = float(row["ftp_w"])
        mass = float(row["weight_kg"]) + float(row["Bike_kg"])
        cd = float(row["Cd"])
        cda = cd * frontal_area_m2(float(row["weight_kg"]), float(row["height_cm"]))
        p_cap = float(cap_fraction) * ftp

        lo, hi = 0.0, 30.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            p = mid * (mass * G * float(crr)) + 0.5 * float(rho) * cda * (mid ** 3)
            if p < p_cap:
                lo = mid
            else:
                hi = mid
        return lo

    sel_edit = sel_edit.copy()
    sel_edit["__order_score_v"] = sel_edit.apply(_solve_front_speed_mps, axis=1)
    sel_edit = sel_edit.sort_values("__order_score_v", ascending=False).drop(columns="__order_score_v").reset_index(drop=True)

    st.caption("Rider order is enforced strongest â weakest (based on cap-speed proxy using FTP, CdA, and mass).")
    st.dataframe(sel_edit[["name","height_cm","weight_kg","ftp_w","Bike","Bike_kg","Cd"]], use_container_width=True, hide_index=True)

    # Draft model editor (default rules; for N>4 positions beyond 4 default to pos4 factor)
    st.subheader("Draft model (CdA factors by position)")
    draft_defaults = build_default_draft_factors(len(sel_edit))
    draft_df = pd.DataFrame({"Position": list(range(1, len(sel_edit)+1)), "CdA factor": draft_defaults})
    draft_df = st.data_editor(draft_df, use_container_width=True, hide_index=True, key="draft_df")
    draft_factors = [float(x) for x in draft_df["CdA factor"].tolist()]

    # Build Rider objects for solver
    riders: List[Rider] = []
    for _, row in sel_edit.iterrows():
        riders.append(
            Rider(
                name=str(row["name"]),
                weight_kg=float(row["weight_kg"]),
                height_cm=float(row["height_cm"]),
                ftp_w=float(row["ftp_w"]),
                bike_kg=float(row["Bike_kg"]),
                cd=float(row["Cd"]),
            )
        )

    if st.button("Compute max-speed plan", key="compute_plan"):
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

        st.session_state["plan"] = plan
        st.session_state["riders_for_plan"] = riders
        st.session_state["draft_factors"] = draft_factors
        st.session_state["env"] = {"crr": float(crr), "rho": float(rho), "cap_fraction": float(cap_fraction)}
        # reset manual pulls
        st.session_state.pop("manual_pulls", None)
        st.session_state.pop("combined_table", None)
        st.rerun()

    if "plan" not in st.session_state:
        st.info("Compute a plan to see results.")
        st.stop()

    plan = st.session_state["plan"]
    riders = st.session_state["riders_for_plan"]
    draft_factors = st.session_state["draft_factors"]
    crr = st.session_state["env"]["crr"]
    rho = st.session_state["env"]["rho"]

    st.success(plan.get("note", "Plan computed."))
    st.metric("Target speed (km/h)", int(round(plan["v_kph"])))
    st.metric("Target speed (m/s)", int(round(plan["v_mps"])))

    pulls = plan["pulls_s"].copy()
    P = compute_power_matrix(riders, plan["v_mps"], float(crr), float(rho), draft_factors)
    avgW = avg_power_for_pulls(P, pulls)

    st.subheader("Combined rider plan (starting order)")
    df_combined = build_combined_results_table(riders, pulls, P, avgW)
    st.dataframe(df_combined, use_container_width=True, hide_index=True)

    # Manual pull adjustment
    st.subheader("Manually adjust pull durations (recalculate %FTP etc.)")
    st.caption("Edits keep the same target speed; the combined table updates to reflect new duty shares.")

    manual_df = pd.DataFrame(
        [{"Rider": r.name, "Pull_s": int(round(pulls[i]))} for i, r in enumerate(riders)]
    )
    manual_edit = st.data_editor(
        manual_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rider": st.column_config.TextColumn(disabled=True),
            "Pull_s": st.column_config.NumberColumn(step=1, format="%d"),
        },
        key="manual_pulls_editor",
    )

    if st.button("Recalculate with manual pulls", key="recalc_manual"):
        new_pulls = pulls.copy()
        for i in range(len(riders)):
            new_pulls[i] = float(manual_edit.loc[i, "Pull_s"])
        if new_pulls.sum() <= 0:
            st.warning("Total rotation time must be > 0.")
        else:
            avgW2 = avg_power_for_pulls(P, new_pulls)
            df_new = build_combined_results_table(riders, new_pulls, P, avgW2)
            st.session_state["manual_pulls"] = new_pulls
            st.session_state["combined_table"] = df_new
            st.rerun()

    if "combined_table" in st.session_state and "manual_pulls" in st.session_state:
        st.subheader("Combined rider plan (after manual pull edits)")
        st.dataframe(st.session_state["combined_table"], use_container_width=True, hide_index=True)

    st.subheader("Power required by position (whole numbers)")
    df_pos = build_position_power_table(riders, plan["v_mps"], float(crr), float(rho), draft_factors)
    st.dataframe(df_pos, use_container_width=True, hide_index=True)
