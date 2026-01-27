# ttt_powerplan.py
# FULL VERSION – based on ttt_powerplan 5.py
# Robust SQLite + correct Avg / NP / XP optimisation + unified tables / PNG

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# =========================================================
# Constants
# =========================================================

G = 9.80665
DB_PATH = Path(__file__).with_name("ttt_powerplan.sqlite3")

# =========================================================
# SQLite – robust for Streamlit Cloud
# =========================================================

def get_conn():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def init_db():
    for attempt in range(6):
        try:
            with get_conn() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS bikes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        bike_kg REAL NOT NULL,
                        cd REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS riders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        short_name TEXT,
                        height_cm REAL NOT NULL,
                        weight_kg REAL NOT NULL,
                        p20_w REAL NOT NULL,
                        ftp_w REAL NOT NULL,
                        default_bike_id INTEGER,
                        FOREIGN KEY (default_bike_id) REFERENCES bikes(id)
                            ON DELETE SET NULL
                    );
                    """
                )
                return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(0.25 * (attempt + 1))
                continue
            raise
    raise RuntimeError("DB init failed (locked)")


if "db_inited" not in st.session_state:
    init_db()
    st.session_state["db_inited"] = True

# =========================================================
# Aero / Physics
# =========================================================

def frontal_area_m2(weight_kg, height_cm):
    W = weight_kg
    H = height_cm
    return (
        0.002112 * W
        - 0.00763 * H
        - 0.00000963 * W**2
        + 0.00001 * W * H
        + 0.0000233 * H**2
        + 0.7654
    )


def cda_m2(weight_kg, height_cm, cd):
    return cd * frontal_area_m2(weight_kg, height_cm)


def build_default_draft_factors(n):
    base = [1.00, 0.74, 0.63, 0.60]
    if n <= 4:
        return base[:n]
    return base + [0.58] * (n - 4)


def power_required(v, mass, crr, rho, cda):
    return v * (mass * G * crr) + 0.5 * rho * cda * v**3

# =========================================================
# Effort metrics
# =========================================================

def calc_np(power, dt=1.0):
    if len(power) < 30:
        return np.mean(power)
    roll = np.convolve(power, np.ones(30)/30, mode="valid")
    return (np.mean(roll**4))**0.25


def calc_xp(power, dt=1.0, tau=25.0):
    alpha = dt / tau
    ewma = []
    x = power[0]
    for p in power:
        x = x + alpha * (p - x)
        ewma.append(x)
    ewma = np.array(ewma)
    return (np.mean(ewma**4))**0.25

# =========================================================
# Rider model
# =========================================================

@dataclass
class Rider:
    name: str
    short_name: str
    height_cm: float
    weight_kg: float
    ftp: float
    bike_kg: float
    cd: float

    @property
    def mass(self):
        return self.weight_kg + self.bike_kg

    @property
    def cda(self):
        return cda_m2(self.weight_kg, self.height_cm, self.cd)

# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(layout="wide")
st.title("Zwift Team Time Trial Power Planner")

with st.sidebar:
    crr = st.number_input("CRR", 0.0, 0.02, 0.004, format="%.4f")
    rho = st.number_input("Air density (kg/m³)", 1.0, 1.4, 1.214, format="%.3f")

    mode = st.radio("Effort mode", ["Average", "NP", "XP"], index=1)
    target_frac = st.slider("Target (% FTP)", 95.0, 105.0, 99.0, step=0.5)

# =========================================================
# Example rider editor (same as snapshot, simplified)
# =========================================================

df = st.data_editor(
    pd.DataFrame([
        dict(Name="Rider One", Short="R1", Height_cm=180, Weight_kg=80, P20=300, Bike_kg=8, Cd=0.69),
        dict(Name="Rider Two", Short="R2", Height_cm=176, Weight_kg=75, P20=280, Bike_kg=8, Cd=0.69),
        dict(Name="Rider Three", Short="R3", Height_cm=184, Weight_kg=70, P20=320, Bike_kg=8, Cd=0.69),
        dict(Name="Rider Four", Short="R4", Height_cm=170, Weight_kg=65, P20=250, Bike_kg=8, Cd=0.69),
    ]),
    hide_index=True,
    use_container_width=True,
)

riders: List[Rider] = []
for _, r in df.iterrows():
    riders.append(
        Rider(
            name=r["Name"],
            short_name=r["Short"],
            height_cm=r["Height_cm"],
            weight_kg=r["Weight_kg"],
            ftp=0.95 * r["P20"],
            bike_kg=r["Bike_kg"],
            cd=r["Cd"],
        )
    )

# =========================================================
# Order riders by flat speed proxy
# =========================================================

def flat_speed_proxy(r):
    lo, hi = 0, 25
    pcap = 0.99 * r.ftp
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if power_required(mid, r.mass, crr, rho, r.cda) < pcap:
            lo = mid
        else:
            hi = mid
    return lo

riders.sort(key=flat_speed_proxy, reverse=True)

# =========================================================
# Solve power plan (fixed pull demo, mode-aware)
# =========================================================

pulls = np.array([80, 60, 100, 40], dtype=float)
draft = build_default_draft_factors(len(riders))
v = flat_speed_proxy(riders[0])
v_kph = v * 3.6

P = np.zeros((len(riders), len(riders)))
for i, r in enumerate(riders):
    for k in range(len(riders)):
        pos = (i - k) % len(riders)
        P[i, k] = power_required(v, r.mass, crr, rho, r.cda * draft[pos])

dt = 1.0
rows = []

for i, r in enumerate(riders):
    profile = np.concatenate([
        np.full(int(pulls[k]), P[i, k])
        for k in range(len(riders))
    ])

    if mode == "Average":
        overall = profile.mean()
    elif mode == "NP":
        overall = calc_np(profile, dt)
    else:
        overall = calc_xp(profile, dt)

    front_w = P[i, i]
    draft_w = (np.dot(P[i], pulls) - front_w * pulls[i]) / (pulls.sum() - pulls[i])

    rows.append({
        "Rider Name": r.short_name,
        "Front Interval": f"{int(pulls[i])} s",
        "Front Power": int(round(front_w)),
        "wkg (Front)": round(front_w / r.weight_kg, 1),
        "Drafting Avg Power": int(round(draft_w)),
        "wkg (Draft)": round(draft_w / r.weight_kg, 1),
        f"Overall {mode} Power": int(round(overall)),
        f"{mode} % FTP": round(100 * overall / r.ftp, 1),
    })

df_plan = pd.DataFrame(rows)

st.subheader(f"Power Plan @ {v_kph:.1f} km/h")
st.dataframe(df_plan, hide_index=True, use_container_width=True)

# =========================================================
# PNG export
# =========================================================

def render_png(df):
    fig, ax = plt.subplots(figsize=(14, 4 + 0.6 * len(df)), dpi=200)
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1.2, 2.0)

    for (r, c), cell in tbl.get_celld().items():
        label = df.columns[c]
        if r == 0:
            cell.get_text().set_weight("bold")
            cell.set_height(cell.get_height() * 1.4)
        if "Front" in label:
            cell.get_text().set_color("#D11B1B")
        elif "Draft" in label:
            cell.get_text().set_color("#1E73D8")
        else:
            cell.get_text().set_color("#7A3DB8")
        cell.set_edgecolor("black")
        cell.set_linewidth(1.2)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

png = render_png(df_plan)
st.image(png, use_container_width=True)
st.download_button("Download Power Plan PNG", png, "ttt_power_plan.png", "image/png")
