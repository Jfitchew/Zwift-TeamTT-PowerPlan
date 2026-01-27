# ttt_powerplan.py
# Streamlit app: Zwift TTT pull & power planner + local rider database (SQLite)
#
# Run:
#   pip install -r requirements.txt
#   streamlit run ttt_powerplan.py

from __future__ import annotations

import math
import sqlite3
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

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
      pos1=1.00, pos2=0.74, pos3=0.63, pos4=0.60
    For n>4 we cap at 0.58 for deeper positions by default.
    """
    base = [1.00, 0.74, 0.63, 0.60]
    if n_riders <= 4:
        return base[:n_riders]
    return base + [0.58] * (n_riders - 4)


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
# Effort metrics (Average / NP / XP)
# =============================
def compute_np_w(power_w: np.ndarray, window_s: int = 30) -> float:
    """TrainingPeaks-style Normalized Power (NP): 30 s rolling average, 4th-power mean."""
    p = np.asarray(power_w, dtype=float)
    if p.size == 0:
        return 0.0
    w = int(window_s)
    if p.size < w:
        # If shorter than window, NP collapses to mean power
        return float(p.mean())

    # Pad start to avoid dropping early samples (simple, stable for periodic series)
    pad = np.full(w - 1, p[0], dtype=float)
    pp = np.concatenate([pad, p])

    c = np.cumsum(pp)
    roll = (c[w:] - c[:-w]) / w  # length == p.size
    return float(np.mean(roll**4) ** 0.25)


def compute_xp_w(power_w: np.ndarray, tau_s: float = 25.0) -> float:
    """Skiba/TrainingPeaks-style xPower (XP): EWMA with ~25 s time constant, 4th-power mean."""
    p = np.asarray(power_w, dtype=float)
    if p.size == 0:
        return 0.0
    if p.size == 1:
        return float(p[0])

    # 1 Hz sample
    alpha = 1.0 - math.exp(-1.0 / float(tau_s))
    ew = np.empty_like(p, dtype=float)
    ew[0] = p[0]
    for i in range(1, p.size):
        ew[i] = ew[i - 1] + alpha * (p[i] - ew[i - 1])

    return float(np.mean(ew**4) ** 0.25)


def build_cycle_power_series_1hz(P_row: np.ndarray, pulls_s: np.ndarray) -> np.ndarray:
    """Piecewise-constant power series (1 Hz) for one full rotation."""
    parts = []
    for k in range(len(pulls_s)):
        dur = int(round(float(pulls_s[k])))
        if dur <= 0:
            continue
        parts.append(np.full(dur, float(P_row[k]), dtype=float))
    if not parts:
        return np.zeros(0, dtype=float)
    return np.concatenate(parts)


def compute_effort_w_for_pulls(P: np.ndarray, pulls_s: np.ndarray, method: str) -> np.ndarray:
    """Return per-rider effort metric over a periodic rotation: Average / NP / XP."""
    method = (method or "Average").strip().upper()
    n = P.shape[0]
    out = np.zeros(n, dtype=float)

    # Use a periodic repetition to reduce boundary effects of NP/XP smoothing filters.
    # (Much faster than simulating a full hour and accurate for steady rotating TTT pacing.)
    for i in range(n):
        cycle = build_cycle_power_series_1hz(P[i, :], pulls_s)
        if cycle.size == 0:
            out[i] = 0.0
            continue

        if method == "AVERAGE":
            out[i] = float(cycle.mean())
            continue

        # Repeat enough to warm up filters and capture steady-state
        L = cycle.size
        reps = max(8, int(math.ceil(240.0 / L)) + 2)  # at least ~4 min total
        series = np.tile(cycle, reps)
        warm = min(series.size - 1, 2 * L)  # discard first 2 cycles

        s = series[warm:]
        if method == "NP":
            out[i] = compute_np_w(s, window_s=30)
        elif method in ("XP", "XPOWER"):
            out[i] = compute_xp_w(s, tau_s=25.0)
        else:
            # Fallback
            out[i] = float(cycle.mean())


def balance_pulls_to_targets(
    P: np.ndarray,
    pulls: np.ndarray,
    method: str,
    ftps: np.ndarray,
    cap_fraction: float,
    strongest_pull_bounds: Tuple[float, float],
    other_pull_max_s: float,
    min_pull_s: float,
    allow_zero_pull: bool,
    iters: int = 600,
    step_s: float = 1.0,
) -> np.ndarray:
    """Heuristic rebalancing at fixed speed.

    Moves small amounts of pull time from the most 'strained' rider (highest metric/target)
    to the least strained rider, while keeping total rotation time constant and respecting
    pull duration bounds. This tends to bring all riders' selected metric (Avg/NP/XP) closer
    to cap_fraction * FTP, improving fairness and making the plan better match the chosen mode.
    """
    n = len(pulls)
    if n <= 1:
        return pulls

    pulls = pulls.astype(float).copy()
    tmin = 0.0 if allow_zero_pull else float(min_pull_s)

    def clamp(i: int) -> None:
        if i == 0:
            pulls[i] = float(np.clip(pulls[i], strongest_pull_bounds[0], strongest_pull_bounds[1]))
        else:
            pulls[i] = float(np.clip(pulls[i], tmin, float(other_pull_max_s)))

    for i in range(n):
        clamp(i)

    target = cap_fraction * ftps

    def score(eff: np.ndarray) -> float:
        # mean absolute relative error to target
        rel = eff / np.maximum(1e-9, target)
        return float(np.mean(np.abs(rel - 1.0)))

    eff = compute_effort_w_for_pulls(P, pulls, method)
    best_score = score(eff)

    for _ in range(iters):
        eff = compute_effort_w_for_pulls(P, pulls, method)
        rel = eff / np.maximum(1e-9, target)

        i_hi = int(np.argmax(rel))
        i_lo = int(np.argmin(rel))

        if i_hi == i_lo:
            break

        # If already reasonably close, stop.
        if best_score <= 0.003:  # ~0.3% mean error
            break

        # Determine if we can move time.
        hi_min = strongest_pull_bounds[0] if i_hi == 0 else tmin
        lo_max = strongest_pull_bounds[1] if i_lo == 0 else float(other_pull_max_s)

        if pulls[i_hi] <= hi_min + 1e-9 or pulls[i_lo] >= lo_max - 1e-9:
            break

        delta = min(float(step_s), pulls[i_hi] - hi_min, lo_max - pulls[i_lo])
        if delta <= 1e-9:
            break

        # Propose move: take from hi, give to lo
        pulls[i_hi] -= delta
        pulls[i_lo] += delta
        clamp(i_hi)
        clamp(i_lo)

        eff2 = compute_effort_w_for_pulls(P, pulls, method)

        # Keep feasibility: do not exceed target by more than tiny tolerance
        if np.any(eff2 - target > 0.5):  # allow small numerical wobble, 0.5 W headroom
            # revert
            pulls[i_hi] += delta
            pulls[i_lo] -= delta
            clamp(i_hi)
            clamp(i_lo)
            # reduce step gradually
            step_s = max(0.25, step_s * 0.7)
            continue

        s2 = score(eff2)
        if s2 + 1e-9 < best_score:
            best_score = s2
        else:
            # revert if no improvement
            pulls[i_hi] += delta
            pulls[i_lo] -= delta
            clamp(i_hi)
            clamp(i_lo)
            step_s = max(0.25, step_s * 0.7)

    return pulls

    return out

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
    effort_method: str,
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
        effortW = compute_effort_w_for_pulls(P, pulls, effort_method)
        viol = effortW - caps
        if np.all(viol <= 1e-6):
            # Optional balancing: redistribute pull time to bring riders' chosen metric (Avg/NP/XP)
            # closer to the target cap_fraction across the whole team, not just <= cap.
            pulls = balance_pulls_to_targets(
                P=P,
                pulls=pulls,
                method=effort_method,
                ftps=ftps,
                cap_fraction=cap_fraction,
                strongest_pull_bounds=strongest_pull_bounds,
                other_pull_max_s=other_pull_max_s,
                min_pull_s=min_pull_s,
                allow_zero_pull=allow_zero_pull,
                iters=600,
                step_s=1.0,
            )
            effortW = compute_effort_w_for_pulls(P, pulls, effort_method)
            return True, pulls, effortW, effortW / ftps

        i_bad = int(np.argmax(viol))
        min_allowed = strongest_pull_bounds[0] if i_bad == 0 else tmin
        if pulls[i_bad] <= min_allowed + 1e-9:
            return False, pulls, effortW, effortW / ftps

        delta = min(2.0, pulls[i_bad] - min_allowed)  # seconds
        pulls[i_bad] -= delta
        clamp(i_bad)

        # Recompute slack, then assign delta to best receiver.
        effortW2 = compute_effort_w_for_pulls(P, pulls, effort_method)
        frac2 = effortW2 / ftps
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
            return False, pulls, effortW2, frac2

        _, j_best = max(candidates, key=lambda x: x[0])
        pulls[j_best] += delta
        clamp(j_best)

    effortW = compute_effort_w_for_pulls(P, pulls, effort_method)
    return bool(np.all(effortW <= caps + 1e-3)), pulls, effortW, effortW / ftps


def search_max_speed_plan(
    riders: List[Rider],
    crr: float,
    rho: float,
    draft_factors: List[float],
    effort_method: str,
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
            effort_method=effort_method,
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
            effort_method=effort_method,
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
            effort_method=effort_method,
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
def build_combined_results_table(riders: List[Rider], pulls: np.ndarray, P: np.ndarray, avgW: np.ndarray, effortW: np.ndarray, effort_method: str) -> pd.DataFrame:
    """
    Combined table (starting order):
      - Pull time
      - Pull (front) power
      - Avg drafting power (time-weighted over non-front segments)
      - Overall avg power
      - %FTP
    Displayed as whole numbers.
    """
    effort_key = {'NP':'NP','XP':'XP','Average':'AVG'}.get(effort_method, (effort_method or 'EFF').upper())
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
                "Rider Name": (r.short_name if getattr(r, "short_name", None) else r.name),
                "Pull_s": int(round(t_front)),
                "Pull_W": int(round(p_front)),
                "DraftAvg_W": int(round(p_draft_avg)),
                "Avg_W": int(round(float(avgW[i]))),
                f"{effort_key}_W": int(round(float(effortW[i]))),
                f"{effort_key}_%FTP": round(100.0 * float(effortW[i]) / float(r.ftp_w), 1),
            }
        )
    return pd.DataFrame(rows)


def build_position_power_table(riders: List[Rider], v_mps: float, crr: float, rho: float, draft_factors: List[float]) -> pd.DataFrame:
    n = len(riders)
    rows = []
    for r in riders:
        row = {"Rider Name": (r.short_name if getattr(r, "short_name", None) else r.name), "FTP_W": int(round(r.ftp_w))}
        for pos in range(1, n + 1):
            cda_eff = r.cda_front * float(draft_factors[pos - 1])
            row[f"Pos{pos}_W"] = int(round(power_required_w(v_mps, r.system_mass_kg, crr, rho, cda_eff)))
        rows.append(row)
    return pd.DataFrame(rows)

def plan_table_png(df: pd.DataFrame, font_size: int = 18) -> bytes:
    """
    Render the presentation 'card' table to PNG bytes with:
      - wide, readable layout (no squashing)
      - 2-line headers (provided via \n in df.columns)
      - per-column coloured text (Front=red, Draft=blue, Overall/%FTP=purple)
      - generous header height so wrapped titles don't collide
    Returns PNG bytes suitable for st.download_button(...).
    """
    # Colour palette (approx to your reference)
    RED = "#D11B1B"
    BLUE = "#1E73D8"
    PURPLE = "#7A3DB8"
    BLACK = "#000000"

    cols = list(df.columns)
    nrows, ncols = df.shape

    def _clean(s: str) -> str:
        return str(s).replace("\n", " ").strip().lower()

    def _col_color(col: str) -> str:
        c = _clean(col)
        if c.startswith("rider"):
            return BLACK
        if c.startswith("front"):
            return RED
        if c.startswith("draft"):
            return BLUE
        if c.startswith("overall") or ("% ftp" in c):
            return PURPLE
        return BLACK

    # Compute relative column widths from max text length (header + body), but keep sane limits.
    def _max_line_len(s: str) -> int:
        parts = str(s).split("\n")
        return max(len(p) for p in parts) if parts else len(str(s))

    max_chars = []
    for c in cols:
        header_len = _max_line_len(c)
        body_lens = [_max_line_len(x) for x in df[c].astype(str).tolist()]
        mc = max([header_len] + body_lens)
        max_chars.append(mc)

    max_chars = np.array(max_chars, dtype=float)

    # Bias certain columns wider (names + interval strings)
    for idx, c in enumerate(cols):
        cc = _clean(c)
        if cc.startswith("rider"):
            max_chars[idx] *= 1.35
        if "interval" in cc:
            max_chars[idx] *= 1.15

    # Add breathing room and clamp extremes
    max_chars = np.clip(max_chars * 1.15, 6.0, 40.0)
    col_widths = (max_chars / max_chars.sum()).tolist()

    # Figure sizing: scale with content and font size
    # Wider than before and taller header area.
    fig_w = max(14.0, min(32.0, 0.55 * float(max_chars.sum())))
    fig_h = max(4.2, 1.4 + 0.72 * (nrows + 1))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=cols,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)

    # Global scaling: x slightly, y more for row height
    tbl.scale(1.08, 1.85)

    # Style cells
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#000000")
        cell.set_facecolor("#FFFFFF")
        cell.set_linewidth(1.6)
        cell.PAD = 0.14  # padding inside cells

        # Set text styling
        col_name = cols[c] if c < len(cols) else ""
        color = _col_color(col_name)
        cell.get_text().set_color(color)
        cell.get_text().set_weight("bold")
        cell.get_text().set_va("center")
        cell.get_text().set_ha("center")

        # Header row: make it taller and a touch thicker border
        if r == 0:
            cell.set_linewidth(1.9)
            # Increase header height relative to body rows
            cell.set_height(cell.get_height() * 1.55)

    # Save with extra padding to avoid clipping
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", pad_inches=0.45)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
def db_path() -> Path:
    # Local to the script directory for portability.
    return Path(__file__).with_name("ttt_riders.sqlite3")


def get_conn() -> sqlite3.Connection:
    """SQLite connection configured to be resilient on Streamlit Cloud."""
    conn = sqlite3.connect(
        str(db_path()),
        timeout=30,              # seconds
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row

    # Pragmas for concurrency + safety
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")     # allow readers during writes
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")   # ms

    return conn



def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def init_db() -> None:
    """Create tables and migrate schemas. Robust to Streamlit Cloud locking."""
    for attempt in range(6):
        try:
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
                        short_name TEXT,
                        height_cm REAL NOT NULL,
                        weight_kg REAL NOT NULL,
                        p20_w REAL NOT NULL,
                        ftp_w REAL NOT NULL,
                        effective_max_hr REAL,
                        strava_url TEXT,
                        zwiftpower_url TEXT,
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
                            short_name TEXT,
                            height_cm REAL NOT NULL,
                            weight_kg REAL NOT NULL,
                            p20_w REAL NOT NULL,
                            ftp_w REAL NOT NULL,
                            effective_max_hr REAL,
                            strava_url TEXT,
                            zwiftpower_url TEXT,
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
                            INSERT OR REPLACE INTO riders_new(name, short_name, height_cm, weight_kg, p20_w, ftp_w, effective_max_hr, strava_url, zwiftpower_url, default_bike_id)
                            VALUES (?,?,?,?,?,?,?,?,?,?)
                            """,
                            (r["name"], None, float(r["height_cm"]), float(r["weight_kg"]), float(r["ftp_w"]) / 0.95, float(r["ftp_w"]), None, None, None, bike_id),
                        )

                    # Replace old table
                    conn.execute("DROP TABLE riders")
                    conn.execute("ALTER TABLE riders_new RENAME TO riders")

                # Migration: add extended rider fields and 20-min power based FTP if missing.
                cols = _table_columns(conn, "riders")
                # Add missing columns (SQLite supports ADD COLUMN).
                def _add_col(sql: str):
                    try:
                        conn.execute(sql)
                    except Exception:
                        pass

                if "short_name" not in cols:
                    _add_col("ALTER TABLE riders ADD COLUMN short_name TEXT")
                if "p20_w" not in cols:
                    _add_col("ALTER TABLE riders ADD COLUMN p20_w REAL")
                if "effective_max_hr" not in cols:
                    _add_col("ALTER TABLE riders ADD COLUMN effective_max_hr REAL")
                if "strava_url" not in cols:
                    _add_col("ALTER TABLE riders ADD COLUMN strava_url TEXT")
                if "zwiftpower_url" not in cols:
                    _add_col("ALTER TABLE riders ADD COLUMN zwiftpower_url TEXT")

                cols = _table_columns(conn, "riders")
                # If ftp_w exists but p20_w is null, backfill p20_w = ftp_w / 0.95
                if "ftp_w" in cols and "p20_w" in cols:
                    conn.execute(
                        "UPDATE riders SET p20_w = COALESCE(p20_w, ftp_w / 0.95) WHERE p20_w IS NULL"
                    )
                    # Ensure ftp_w matches 0.95*p20_w going forward
                    conn.execute(
                        "UPDATE riders SET ftp_w = 0.95 * p20_w WHERE p20_w IS NOT NULL"
                    )


                # Ensure at least one bike exists
                n_bikes = conn.execute("SELECT COUNT(*) AS n FROM bikes").fetchone()["n"]
                if n_bikes == 0:
                    conn.execute("INSERT INTO bikes(name, bike_kg, cd) VALUES (?,?,?)", ("Default", 8.0, 0.69))

                conn.commit()
                return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(0.25 * (attempt + 1))
                continue
            raise
    raise RuntimeError("Database initialisation failed (database is locked).")


def ensure_db() -> None:
    """Ensure DB is initialised once per Streamlit session."""
    if st.session_state.get("_db_inited", False):
        return
    ensure_db()
    st.session_state["_db_inited"] = True



def fetch_bikes_df() -> pd.DataFrame:
    ensure_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, bike_kg, cd FROM bikes ORDER BY name ASC"
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["id", "name", "bike_kg", "cd"])
    return pd.DataFrame([dict(r) for r in rows])


def upsert_bike(name: str, bike_kg: float, cd: float) -> None:
    ensure_db()
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
    ensure_db()
    with get_conn() as conn:
        n_bikes = conn.execute("SELECT COUNT(*) AS n FROM bikes").fetchone()["n"]
        if n_bikes <= 1:
            # Keep at least one bike
            return
        conn.execute("DELETE FROM bikes WHERE name=?", (name.strip(),))
        conn.commit()


def fetch_riders_df() -> pd.DataFrame:
    ensure_db()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
              r.id, r.name, r.short_name, r.height_cm, r.weight_kg, r.p20_w, r.ftp_w, r.effective_max_hr, r.strava_url, r.zwiftpower_url,
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
    short_name: str | None,
    height_cm: float,
    weight_kg: float,
    p20_w: float,
    effective_max_hr: float | None,
    strava_url: str | None,
    zwiftpower_url: str | None,
    default_bike_id: int | None,
) -> None:
    """Insert or update a rider. FTP is stored as 0.95 * 20-min power."""
    ensure_db()
    ftp_w = 0.95 * float(p20_w)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO riders(
              name, short_name, height_cm, weight_kg, p20_w, ftp_w,
              effective_max_hr, strava_url, zwiftpower_url, default_bike_id
            )
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
              short_name=excluded.short_name,
              height_cm=excluded.height_cm,
              weight_kg=excluded.weight_kg,
              p20_w=excluded.p20_w,
              ftp_w=excluded.ftp_w,
              effective_max_hr=excluded.effective_max_hr,
              strava_url=excluded.strava_url,
              zwiftpower_url=excluded.zwiftpower_url,
              default_bike_id=excluded.default_bike_id
            """,
            (
                name.strip(),
                (short_name.strip() if isinstance(short_name, str) and short_name.strip() else None),
                float(height_cm),
                float(weight_kg),
                float(p20_w),
                float(ftp_w),
                (float(effective_max_hr) if effective_max_hr not in (None, "") else None),
                (str(strava_url).strip() if isinstance(strava_url, str) and str(strava_url).strip() else None),
                (str(zwiftpower_url).strip() if isinstance(zwiftpower_url, str) and str(zwiftpower_url).strip() else None),
                default_bike_id,
            ),
        )
        conn.commit()


def delete_rider_by_name(name: str) -> None:
    ensure_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM riders WHERE name=?", (name.strip(),))
        conn.commit()


def export_bikes_csv() -> str:
    df = fetch_bikes_df()
    cols = ["name", "bike_kg", "cd"]
    return df[cols].to_csv(index=False)


def export_riders_csv() -> str:
    df = fetch_riders_df()
    cols = [
        "name",
        "short_name",
        "height_cm",
        "weight_kg",
        "p20_w",
        "ftp_w",
        "effective_max_hr",
        "strava_url",
        "zwiftpower_url",
        "default_bike_name",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols).to_csv(index=False)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].copy()
    return df.to_csv(index=False)
def import_bikes_csv(csv_bytes: bytes) -> Tuple[int, List[str]]:
    """Import bikes from CSV with columns: name,bike_kg,cd"""
    ensure_db()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"name", "bike_kg", "cd"}
    if not required.issubset(set(df.columns)):
        return 0, [f"CSV must contain columns: {sorted(required)}"]

    errors: List[str] = []
    count = 0
    for _, r in df.iterrows():
        try:
            upsert_bike(
                name=str(r["name"]).strip(),
                bike_kg=float(r["bike_kg"]),
                cd=float(r["cd"]),
            )
            count += 1
        except Exception as e:
            errors.append(f"{r.get('name','<blank>')}: {e}")
    return count, errors


def _normalize_riders_import_df(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts common column headings from CSV/XLSX and returns normalized columns."""
    # Map common headers (case-insensitive)
    colmap = {c.lower().strip(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n.lower() in colmap:
                return colmap[n.lower()]
        return None

    name_c = pick("name", "rider", "rider name")
    height_c = pick("height_cm", "height (cm)", "height", "Height (cm)")
    weight_c = pick("weight_kg", "weight (kg)", "weight", "Weight (kg)")
    p20_c = pick("p20_w", "20min power (w)", "20min max power", "20min max power (w)", "20min power", "20min", "20min Power (W)")
    ftp_c = pick("ftp_w", "ftp (w)", "ftp", "FTP (W)")
    short_c = pick("short_name", "short name", "Short Name")
    hr_c = pick("effective_max_hr", "effective maxhr", "effective max hr", "Effective MaxHR")
    strava_c = pick("strava_url", "strava", "Strava")
    zwp_c = pick("zwiftpower_url", "zwift power", "zwiftpower", "Zwift Power")
    bike_c = pick("default_bike_name", "bike", "bike name", "default bike", "default_bike")

    if name_c is None:
        raise ValueError("Missing rider name column (e.g. 'name' or 'Rider').")
    if height_c is None or weight_c is None:
        raise ValueError("Missing height/weight columns (e.g. 'Height (cm)' and 'weight (kg)').")

    out = pd.DataFrame()
    out["name"] = df[name_c].astype(str).str.strip()
    out["height_cm"] = pd.to_numeric(df[height_c], errors="coerce")
    out["weight_kg"] = pd.to_numeric(df[weight_c], errors="coerce")

    # Prefer 20-min power; fall back to FTP if provided.
    if p20_c is not None:
        out["p20_w"] = pd.to_numeric(df[p20_c], errors="coerce")
    else:
        out["p20_w"] = np.nan

    if ftp_c is not None:
        ftp = pd.to_numeric(df[ftp_c], errors="coerce")
    else:
        ftp = np.nan

    # If p20 is missing but ftp exists, infer p20 = ftp/0.95
    out["p20_w"] = out["p20_w"].where(~out["p20_w"].isna(), ftp / 0.95)

    out["short_name"] = df[short_c].astype(str).str.strip() if short_c is not None else None
    out["effective_max_hr"] = pd.to_numeric(df[hr_c], errors="coerce") if hr_c is not None else np.nan
    out["strava_url"] = df[strava_c].astype(str).str.strip() if strava_c is not None else None
    out["zwiftpower_url"] = df[zwp_c].astype(str).str.strip() if zwp_c is not None else None
    out["default_bike_name"] = df[bike_c].astype(str).str.strip() if bike_c is not None else None

    # Clean up 'nan' strings from optional text fields
    for c in ["short_name", "strava_url", "zwiftpower_url", "default_bike_name"]:
        if c in out.columns:
            out[c] = out[c].replace({"nan": None, "NaN": None, "None": None})

    # Drop rows without essential numeric fields
    out = out.dropna(subset=["height_cm", "weight_kg", "p20_w"])
    return out


def import_riders_csv(csv_bytes: bytes) -> Tuple[int, List[str]]:
    """Import riders from CSV. FTP is calculated as 0.95 * 20min power."""
    ensure_db()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df = _normalize_riders_import_df(df)

    bikes = fetch_bikes_df()
    bike_name_to_id = {str(r["name"]): int(r["id"]) for _, r in bikes.iterrows()} if not bikes.empty else {}

    errors: List[str] = []
    count = 0

    for _, r in df.iterrows():
        try:
            bike_id = None
            if r.get("default_bike_name"):
                bn = str(r["default_bike_name"]).strip()
                if bn:
                    if bn not in bike_name_to_id:
                        # Create missing bike with defaults
                        upsert_bike(bn, 8.0, 0.69)
                        bikes2 = fetch_bikes_df()
                        bike_name_to_id = {str(x["name"]): int(x["id"]) for _, x in bikes2.iterrows()}
                    bike_id = bike_name_to_id.get(bn)

            upsert_rider(
                name=str(r["name"]),
                short_name=r.get("short_name"),
                height_cm=float(r["height_cm"]),
                weight_kg=float(r["weight_kg"]),
                p20_w=float(r["p20_w"]),
                effective_max_hr=(None if pd.isna(r.get("effective_max_hr")) else float(r.get("effective_max_hr"))),
                strava_url=r.get("strava_url"),
                zwiftpower_url=r.get("zwiftpower_url"),
                default_bike_id=bike_id,
            )
            count += 1
        except Exception as e:
            errors.append(f"{r.get('name','<blank>')}: {e}")
    return count, errors


def import_riders_xlsx(xlsx_bytes: bytes, sheet_name: str | None = None) -> Tuple[int, List[str]]:
    """Import riders from an Excel file (.xlsx)."""
    ensure_db()
    df = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=sheet_name or 0)
    df = _normalize_riders_import_df(df)
    # Reuse CSV importer logic by converting to records loop
    bikes = fetch_bikes_df()
    bike_name_to_id = {str(r["name"]): int(r["id"]) for _, r in bikes.iterrows()} if not bikes.empty else {}

    errors: List[str] = []
    count = 0
    for _, r in df.iterrows():
        try:
            bike_id = None
            if r.get("default_bike_name"):
                bn = str(r["default_bike_name"]).strip()
                if bn:
                    if bn not in bike_name_to_id:
                        upsert_bike(bn, 8.0, 0.69)
                        bikes2 = fetch_bikes_df()
                        bike_name_to_id = {str(x["name"]): int(x["id"]) for _, x in bikes2.iterrows()}
                    bike_id = bike_name_to_id.get(bn)

            upsert_rider(
                name=str(r["name"]),
                short_name=r.get("short_name"),
                height_cm=float(r["height_cm"]),
                weight_kg=float(r["weight_kg"]),
                p20_w=float(r["p20_w"]),
                effective_max_hr=(None if pd.isna(r.get("effective_max_hr")) else float(r.get("effective_max_hr"))),
                strava_url=r.get("strava_url"),
                zwiftpower_url=r.get("zwiftpower_url"),
                default_bike_id=bike_id,
            )
            count += 1
        except Exception as e:
            errors.append(f"{r.get('name','<blank>')}: {e}")
    return count, errors
st.set_page_config(page_title="Zwift TTT Power Plan", layout="wide")

# Ensure DB schema exists (runs once per session)
ensure_db()

st.title("Zwift Indoor Team Time Trial Pull & Power Planner")

ensure_db()
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
        r_short = st.text_input("Short name", value="")
        r_height = st.number_input("Height (cm)", value=180.0, step=1.0)
        r_weight = st.number_input("Weight (kg)", value=75.0, step=0.5)
        r_p20 = st.number_input("20 min max power (W)", value=300.0, step=5.0)
        st.caption(f"Calculated FTP (95% of 20 min): {int(round(0.95 * r_p20))} W")
        r_maxhr = st.number_input("Effective max HR (bpm)", value=0.0, step=1.0, help="Leave 0 if unknown")
        r_strava = st.text_input("Strava link", value="")
        r_zwp = st.text_input("ZwiftPower link", value="")
        r_bike_name = st.selectbox("Default bike", bike_names, index=0, key="r_default_bike") if bike_names else None

        if st.button("Save rider", key="save_rider"):
            if r_name.strip():
                bike_id = bike_name_to_id.get(r_bike_name) if r_bike_name else None
                upsert_rider(
                    name=r_name.strip(),
                    short_name=(r_short.strip() or None),
                    height_cm=float(r_height),
                    weight_kg=float(r_weight),
                    p20_w=float(r_p20),
                    effective_max_hr=(None if float(r_maxhr) <= 0 else float(r_maxhr)),
                    strava_url=(r_strava.strip() or None),
                    zwiftpower_url=(r_zwp.strip() or None),
                    default_bike_id=bike_id,
                )
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

    st.subheader("Import / Export riders (CSV / XLSX)")
    exp = export_riders_csv()
    st.download_button("Download riders.csv", data=exp, file_name="riders.csv", mime="text/csv")

    up = st.file_uploader("Import riders file", type=["csv", "xlsx"], key="import_riders")
    if up is not None:
        fn = (up.name or "").lower()
        if fn.endswith(".xlsx"):
            n_ok, errors = import_riders_xlsx(up.read())
        else:
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
        rho = st.number_input("Air density ÃÂÃÂ (kg/mÃÂÃÂ³)", value=1.214, step=0.01, format="%.3f")

        st.subheader("Constraints")
        effort_method = st.selectbox("Effort metric (constraint & reporting)", ["NP", "XP", "Average"], index=0)
        cap_fraction = st.slider(f"Target {effort_method} (%FTP)", min_value=0.70, max_value=1.05, value=0.99, step=0.01)

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
        "Pick riders (4ÃÂ¢ÃÂÃÂ8). You can edit values temporarily below before solving.",
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
        # Ensure we have 20-min power; if missing, infer from FTP.
    sel["p20_w"] = sel["p20_w"].where(~sel["p20_w"].isna(), sel["ftp_w"] / 0.95)
    sel["ftp_w"] = 0.95 * sel["p20_w"]

    st.caption("Edit selected rider values (temporary overrides)")
    sel_edit = st.data_editor(
        sel[["name","height_cm","weight_kg","p20_w","ftp_w","Bike","Bike_kg","Cd"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Rider", disabled=True),
            "height_cm": st.column_config.NumberColumn("Height (cm)", step=1, format="%.0f"),
            "weight_kg": st.column_config.NumberColumn("Weight (kg)", step=0.1, format="%.1f"),
            "p20_w": st.column_config.NumberColumn("20 min max (W)", step=1, format="%.0f"),
            "ftp_w": st.column_config.NumberColumn("FTP (calc, W)", disabled=True, format="%.0f"),
            "Bike": st.column_config.SelectboxColumn("Bike", options=bike_names),
            "Bike_kg": st.column_config.NumberColumn("Bike kg (override)", step=0.1, format="%.1f"),
            "Cd": st.column_config.NumberColumn("Cd (override)", step=0.001, format="%.3f"),
        },
        key="sel_edit",
    )

    # Recompute FTP from edited 20-min power
    sel_edit["ftp_w"] = 0.95 * sel_edit["p20_w"]

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

    st.caption("Rider order is enforced strongest ÃÂ¢ÃÂÃÂ weakest (based on cap-speed proxy using FTP, CdA, and mass).")
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
            effort_method=str(effort_method),
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
        st.session_state["env"] = {"crr": float(crr), "rho": float(rho), "cap_fraction": float(cap_fraction), "effort_method": str(effort_method)}
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
    st.metric("Target speed (km/h)", f"{plan['v_kph']:.1f}")

    pulls = plan["pulls_s"].copy()
    P = compute_power_matrix(riders, plan["v_mps"], float(crr), float(rho), draft_factors)
    avgW = avg_power_for_pulls(P, pulls)
    effort_method = st.session_state["env"].get("effort_method", "NP")
    effortW = compute_effort_w_for_pulls(P, pulls, effort_method)

    st.subheader("Power plan (table)")
    df_combined = build_combined_results_table(riders, pulls, P, avgW, effortW, effort_method)
    # Presentation table (matches your preferred layout) + PNG export
    st.subheader("Power plan card (export)")
    st.caption("This is a compact, shareable table view (PNG), using the current pulls (manual edits included if applied).")

    df_for_card = df_combined
    if "combined_table" in st.session_state and "manual_pulls" in st.session_state:
        df_for_card = st.session_state["combined_table"]

    # Build card-style table
    # Expect df_for_card has columns: Rider, Pull_s, Pull_W, DraftAvg_W, and NP_W/XP_W + NP_%FTP/XP_%FTP
    effort_col_w = None
    effort_col_pct = None
    for c in df_for_card.columns:
        if c.endswith("_W") and c not in ("Avg_W", "Pull_W", "DraftAvg_W"):
            effort_col_w = c
        if c.endswith("_%FTP"):
            effort_col_pct = c

    if effort_col_w is None or effort_col_pct is None:
        st.warning("Card export unavailable: could not find effort columns in the combined table.")
    else:
        effort_method_label = {"NP":"NP","XP":"XP","Average":"Avg"}.get(effort_method, str(effort_method))
        df_card = pd.DataFrame({
            "Rider\nName": df_for_card["Rider Name"].astype(str),
            "Front\nInterval": df_for_card["Pull_s"].astype(int).astype(str) + " secs",
            "Front\nPower": df_for_card["Pull_W"].astype(int),
            "Front\nwkg": (df_for_card["Pull_W"].astype(float) / np.maximum(1e-9, np.array([r.weight_kg for r in riders], dtype=float))).round(1),
            "Drafting\nAvg Power": df_for_card["DraftAvg_W"].astype(int),
            "Drafting\nwkg": (df_for_card["DraftAvg_W"].astype(float) / np.maximum(1e-9, np.array([r.weight_kg for r in riders], dtype=float))).round(1),
            f"Overall\n{effort_method_label} Power": df_for_card[effort_col_w].astype(int),
            f"{effort_method_label}\n% FTP": df_for_card[effort_col_pct].astype(float).round(1),
        })

        # Show the same fields as the PNG card in a normal table as well
        df_display = df_card.copy()
        df_display.columns = [c.replace("\n", " ") for c in df_display.columns]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        png_bytes = plan_table_png(df_card)

        st.image(png_bytes, use_container_width=True)
        st.download_button(
            "Download power plan card (PNG)",
            data=png_bytes,
            file_name="ttt_power_plan.png",
            mime="image/png",
        )
        st.download_button(
            "Download power plan table (CSV)",
            data=df_card.to_csv(index=False).encode("utf-8"),
            file_name="ttt_power_plan.csv",
            mime="text/csv",
        )


    # Manual pull adjustment
    st.subheader("Manually adjust pull durations (recalculate %FTP etc.)")
    st.caption("Edits keep the same target speed; the combined table updates to reflect new duty shares.")

    manual_df = pd.DataFrame(
        [{"Rider Name": (r.short_name if getattr(r, "short_name", None) else r.name), "Pull_s": int(round(pulls[i]))} for i, r in enumerate(riders)]
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
            effortW2 = compute_effort_w_for_pulls(P, new_pulls, effort_method)
            df_new = build_combined_results_table(riders, new_pulls, P, avgW2, effortW2, effort_method)
            st.session_state["manual_pulls"] = new_pulls
            st.session_state["combined_table"] = df_new
            st.rerun()

    if "combined_table" in st.session_state and "manual_pulls" in st.session_state:
        st.subheader("Combined rider plan (after manual pull edits)")
        st.dataframe(st.session_state["combined_table"], use_container_width=True, hide_index=True)

    st.subheader("Power required by position (whole numbers)")
    df_pos = build_position_power_table(riders, plan["v_mps"], float(crr), float(rho), draft_factors)
    st.dataframe(df_pos, use_container_width=True, hide_index=True)
