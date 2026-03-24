#!/usr/bin/env python3
# Copyright (C) 2024 Andrew
# SPDX-License-Identifier: GPL-3.0-or-later
"""SHMIP Suite F: seasonal forcing on valley glacier (T, melt, N bands, Q).

Usage: cd shmip_F && python plot_shmip.py
Output: outputs/shmip_F.png
"""

import sys
import os

sys.argv = sys.argv[:1]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DIR = "outputs/csv"
OUTFILE = "outputs/shmip_F.png"

Lx = 6000.0

CASES   = ["F1", "F2", "F3", "F4", "F5"]
F_MAP   = {"F1": -6.0, "F2": -3.0, "F3": 0.0, "F4": +3.0, "F5": +6.0}

T_AMPL       = 16.0
T_BASE       = -5.0
LR           = -0.0075
DDF          = 0.01 / 86400
BASAL_RATE   = 7.93e-11
SEC_PER_YEAR = 365 * 86400
SEC_PER_DAY  = 86400

S_MID = 100 * (3000/6000 + (3000 + 200)**0.25 - 200**0.25) + 1

BANDS = [
    ("lower (0-2 km)",  0.0,  2e3),
    ("middle (2-4 km)", 2e3,  4e3),
    ("upper (4-6 km)",  4e3,  6e3),
]

CASE_COLORS = {
    "F1": "#2166ac", "F2": "#74add1", "F3": "#878787",
    "F4": "#f46d43", "F5": "#a50026",
}

MONTH_DAYS  = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

DPI = 220


def air_temp(t_days, DT):
    t_s = t_days * SEC_PER_DAY
    return -T_AMPL * np.cos(2.0 * np.pi * t_s / SEC_PER_YEAR) + T_BASE + DT


def melt_rate(t_days, DT, S_elev=S_MID):
    T_air = air_temp(t_days, DT)
    T_local = T_air + LR * S_elev
    return np.maximum(T_local, 0.0) * DDF + BASAL_RATE


def apply_month_ticks(ax, labelbottom=True):
    ax.set_xlim(0, 365)
    ax.set_xticks(MONTH_DAYS)
    ax.set_xticklabels(MONTH_NAMES if labelbottom else [], fontsize=8)


def load_csv(tag):
    nx_path = os.path.join(CSV_DIR, f"{tag}_Nx_seasonal.csv")
    q_path  = os.path.join(CSV_DIR, f"{tag}_Q_seasonal.csv")

    if not os.path.exists(nx_path):
        return None, None, None, None, None

    nx_data = np.loadtxt(nx_path, delimiter=",", skiprows=1, ndmin=2)
    t_days  = nx_data[:, 0]
    N_xt    = nx_data[:, 1:]

    with open(nx_path) as fh:
        hdr = fh.readline().lstrip("#").strip()
    x_c = np.array([float(col.split("=")[1].rstrip("km"))
                     for col in hdr.split(",")[1:]]) * 1e3

    Q_sh = Q_ch = None
    if os.path.exists(q_path):
        q_data = np.loadtxt(q_path, delimiter=",", skiprows=1, ndmin=2)
        if q_data.ndim == 2 and q_data.shape[1] >= 3:
            Q_sh = q_data[:, 1]
            Q_ch = q_data[:, 2]

    return t_days, x_c, N_xt, Q_sh, Q_ch


csv_data = {}
for tag in CASES:
    result = load_csv(tag)
    if result[0] is not None:
        csv_data[tag] = result
        print(f"  loaded {tag}: {result[2].shape[0]} days x {result[2].shape[1]} bins",
              flush=True)

if not csv_data:
    sys.exit("No CSV data found — run run_shmip_F.py first.")

fig = plt.figure(figsize=(15, 10), layout="constrained")
gs = fig.add_gridspec(3, 6, height_ratios=[1, 1.2, 1])

ax_temp = fig.add_subplot(gs[0, 0:3])
ax_melt = fig.add_subplot(gs[0, 3:6])
ax_N = [fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6])]
ax_Qsh = fig.add_subplot(gs[2, 0:3])
ax_Qch = fig.add_subplot(gs[2, 3:6])

t_cont = np.linspace(0, 365, 1200)
for tag in CASES:
    DT = F_MAP[tag]
    ax_temp.plot(t_cont, air_temp(t_cont, DT), lw=2.0, color=CASE_COLORS[tag],
                 label=f"{tag} (dT={DT:+.0f} C)")
ax_temp.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
ax_temp.set_ylabel("T_air (C)")
ax_temp.legend(fontsize=7, ncol=3, loc="lower right")
ax_temp.set_title("Seasonal air temperature", fontweight="bold", loc="left")
apply_month_ticks(ax_temp, labelbottom=False)
ax_temp.grid(True, alpha=0.2)

for tag in CASES:
    DT = F_MAP[tag]
    m = melt_rate(t_cont, DT) * SEC_PER_DAY * 1e3
    ax_melt.plot(t_cont, m, lw=2.0, color=CASE_COLORS[tag], label=f"{tag}")
ax_melt.set_ylabel("Melt rate (mm/day)")
ax_melt.set_ylim(bottom=0)
ax_melt.legend(fontsize=7, ncol=5, loc="upper right")
ax_melt.set_title(f"Degree-day melt at S = {S_MID:.0f} m", fontweight="bold", loc="left")
apply_month_ticks(ax_melt, labelbottom=False)
ax_melt.grid(True, alpha=0.2)

for bi, (bname, x_lo, x_hi) in enumerate(BANDS):
    ax = ax_N[bi]
    for tag in CASES:
        if tag not in csv_data:
            continue
        t_days, x_c, N_xt, _, _ = csv_data[tag]
        mask = (x_c >= x_lo) & (x_c < x_hi)
        if not mask.any():
            continue
        N_mean = N_xt[:, mask].mean(axis=1) / 1e6
        ax.plot(t_days, N_mean, lw=1.8, color=CASE_COLORS[tag],
                label=f"{tag}" if bi == 0 else None)

    ax.set_ylim(bottom=0)
    ax.set_title(f"N_bar — {bname}", fontweight="bold", loc="left", fontsize=10)
    apply_month_ticks(ax, labelbottom=False)
    ax.grid(True, alpha=0.2)
    if bi == 0:
        ax.set_ylabel("N_bar (MPa)")
        ax.legend(fontsize=7, ncol=5, loc="upper right")

for tag in CASES:
    if tag not in csv_data or csv_data[tag][3] is None:
        continue
    t_days, _, _, Q_sh, _ = csv_data[tag]
    ax_Qsh.plot(t_days, Q_sh, lw=1.8, color=CASE_COLORS[tag], label=f"{tag}")
ax_Qsh.set_ylim(bottom=0)
ax_Qsh.set_ylabel("Q_sh (m^3/s)")
ax_Qsh.set_title("Sheet outflow", fontweight="bold", loc="left")
apply_month_ticks(ax_Qsh)
ax_Qsh.grid(True, alpha=0.2)
ax_Qsh.legend(fontsize=7, ncol=5, loc="upper right")

for tag in CASES:
    if tag not in csv_data or csv_data[tag][4] is None:
        continue
    t_days, _, _, _, Q_ch = csv_data[tag]
    Q_ch = np.clip(np.nan_to_num(Q_ch, nan=0.0, posinf=0.0), 0.0, None)
    ax_Qch.plot(t_days, Q_ch, lw=1.8, color=CASE_COLORS[tag], label=f"{tag}")
ax_Qch.set_ylim(bottom=0)
ax_Qch.set_ylabel("Sum|Q_ch| (m^3/s)")
ax_Qch.set_title("Channel flow", fontweight="bold", loc="left")
apply_month_ticks(ax_Qch)
ax_Qch.grid(True, alpha=0.2)
ax_Qch.legend(fontsize=7, ncol=5, loc="upper right")

fig.suptitle(
    "SHMIP Suite F — Seasonal forcing on valley glacier\n"
    "F1-F5: dT = -6, -3, 0, +3, +6 C  |  gamma = 0.05 (Bench Glacier)",
    fontsize=12,
)

os.makedirs(os.path.dirname(OUTFILE) or ".", exist_ok=True)
fig.savefig(OUTFILE, dpi=DPI, bbox_inches="tight")
print(f"Saved → {OUTFILE}")
