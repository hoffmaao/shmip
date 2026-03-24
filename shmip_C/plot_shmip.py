#!/usr/bin/env python3
# Copyright (C) 2024 Andrew
# SPDX-License-Identifier: GPL-3.0-or-later
"""SHMIP Suite C (diurnal moulin): Q(t), N(t) bands, and phase lag.

Usage: cd shmip_C && python plot_shmip.py
Output: outputs/shmip_C.png
"""

import sys
import os

sys.argv = sys.argv[:1]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CSV_DIR = "outputs/csv"
OUTFILE = "outputs/shmip_C.png"

Lx           = 100e3
DOMAIN_AREA  = 2e9
T_HOURS      = 24.0

A1_RATE = 7.93e-11
A5_RATE = 4.5e-8
Q_MEAN  = (A5_RATE - A1_RATE) * DOMAIN_AREA

T_QPEAK_H = 18.0

CASES = {
    "C1": {"ra": 0.25, "color": "#1a9850", "ls": "-"},
    "C2": {"ra": 0.50, "color": "#91cf60", "ls": "--"},
    "C3": {"ra": 1.00, "color": "#fc8d59", "ls": "-."},
    "C4": {"ra": 2.00, "color": "#d73027", "ls": ":"},
}

BANDS = [
    ("lower (0-25 km)",   0.0,   25e3),
    ("upper (75-100 km)", 75e3, 100e3),
]

LAG_BAND = (0.0, 25e3)

DPI = 220


def forcing_Q(t_hours, ra):
    s = np.sin(2.0 * np.pi * t_hours / T_HOURS)
    return Q_MEAN * np.maximum(0.0, 1.0 - ra * s)


def load_Nx_csv(tag):
    path = os.path.join(CSV_DIR, f"{tag}_Nx_diurnal.csv")
    if not os.path.exists(path):
        return None, None, None
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    t_s = data[:, 0]
    N_xt = data[:, 1:]
    with open(path) as fh:
        hdr = fh.readline().lstrip("#").strip()
    x_c = np.array([float(col.split("=")[1].rstrip("km"))
                    for col in hdr.split(",")[1:]]) * 1e3
    return t_s / 3600.0, x_c, N_xt


def band_mean(N_xt, x_c, x_lo, x_hi):
    mask = (x_c >= x_lo) & (x_c < x_hi)
    return N_xt[:, mask].mean(axis=1)


def compute_lag(t_h, N_xt, x_c, x_lo, x_hi):
    mean = band_mean(N_xt, x_c, x_lo, x_hi)
    mask = (t_h >= 0.0) & (t_h <= T_HOURS)
    t_cyc = t_h[mask]
    N_cyc = mean[mask]
    if len(N_cyc) == 0:
        return np.nan
    t_nmin = t_cyc[np.argmin(N_cyc)]
    lag = t_nmin - T_QPEAK_H
    lag = (lag + T_HOURS / 2) % T_HOURS - T_HOURS / 2
    return lag


def _set_time_axis(ax, last=False):
    ax.set_xlim(0, T_HOURS)
    ax.set_xticks([0, 6, 12, 18, 24])
    if last:
        ax.set_xticklabels(["0 h", "6 h", "12 h", "18 h", "24 h"])
        ax.set_xlabel("Time (h)")
    else:
        ax.set_xticklabels([])
    ax.grid(True, alpha=0.22)


nx_data = {}
for tag in CASES:
    t_h, x_c, N_xt = load_Nx_csv(tag)
    if t_h is not None:
        nx_data[tag] = (t_h, x_c, N_xt)
        print(f"  {tag}: {N_xt.shape[0]} time steps x {N_xt.shape[1]} x-bins", flush=True)

if not nx_data:
    sys.exit("No CSV data found — run run_shmip_C.py first.")

fig, axes = plt.subplots(
    4, 1,
    figsize=(8, 13),
    gridspec_kw={"height_ratios": [1.0, 1.2, 1.2, 1.0]},
    constrained_layout=True,
)

t_plot = np.linspace(0.0, T_HOURS, 1000)

ax = axes[0]
for tag, cfg in CASES.items():
    Q = forcing_Q(t_plot, cfg["ra"])
    ax.plot(t_plot, Q, lw=2.0, color=cfg["color"], ls=cfg["ls"],
            label=f"{tag} (ra = {cfg['ra']})")

ax.axhline(Q_MEAN, color="k", ls=":", lw=0.9, alpha=0.5,
           label=f"Q_mean = {Q_MEAN:.0f} m^3/s")
ax.axvline(T_QPEAK_H, color="gray", ls="--", lw=0.8, alpha=0.5)
_set_time_axis(ax)
ax.set_ylim(bottom=-5)
ax.set_ylabel("Q_moulin (m^3/s)")
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.set_title("Moulin recharge", fontweight="bold", loc="left")

for pi, (band_label, x_lo, x_hi) in enumerate(BANDS):
    ax = axes[pi + 1]

    for tag, cfg in CASES.items():
        if tag not in nx_data:
            continue
        t_h, x_c, N_xt = nx_data[tag]
        mean = band_mean(N_xt, x_c, x_lo, x_hi)
        ax.plot(t_h, mean / 1e6, color=cfg["color"], lw=2.0, ls=cfg["ls"],
                label=f"{tag} (ra = {cfg['ra']})")

    ax.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(T_QPEAK_H, color="gray", ls="--", lw=0.8, alpha=0.5)
    _set_time_axis(ax)
    ax.set_ylabel("N_bar (MPa)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title(f"Effective pressure — {band_label}",
                 fontweight="bold", loc="left")

ax = axes[3]
ra_vals = []
lag_vals = []
col_vals = []

for tag, cfg in CASES.items():
    if tag not in nx_data:
        continue
    t_h, x_c, N_xt = nx_data[tag]
    lag = compute_lag(t_h, N_xt, x_c, *LAG_BAND)
    ra_vals.append(cfg["ra"])
    lag_vals.append(lag)
    col_vals.append(cfg["color"])
    print(f"  {tag}: ra = {cfg['ra']:.2f},  lag = {lag:+.2f} h", flush=True)

if ra_vals:
    for ra, lag, color in zip(ra_vals, lag_vals, col_vals):
        ax.scatter(ra, lag, s=80, color=color, zorder=5)
    order = np.argsort(ra_vals)
    ax.plot(np.array(ra_vals)[order], np.array(lag_vals)[order],
            color="black", lw=1.2, ls="-", alpha=0.5, zorder=4)
    for ra, lag, tag in zip(ra_vals, lag_vals,
                            [t for t in CASES if t in nx_data]):
        ax.annotate(tag, (ra, lag), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

ax.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.5)
ax.set_xlabel("Relative amplitude ra")
ax.set_ylabel("Phase lag dt (h)")
ax.grid(True, alpha=0.22)
ax.set_title("N_bar phase lag vs forcing amplitude (lower band 0-25 km)",
             fontweight="bold", loc="left")

fig.suptitle(
    "SHMIP Suite C — Diurnal oscillating moulin recharge\n"
    f"100 moulins, warm-start from B5  |  T = 24 h",
    fontsize=10,
)

os.makedirs(os.path.dirname(OUTFILE) or ".", exist_ok=True)
fig.savefig(OUTFILE, dpi=DPI, bbox_inches="tight")
print(f"Saved → {OUTFILE}")
