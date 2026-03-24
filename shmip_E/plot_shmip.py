#!/usr/bin/env python3
# Copyright (C) 2024 Andrew
# SPDX-License-Identifier: GPL-3.0-or-later
"""SHMIP Suite E ribbon figure from saved checkpoints.

Usage: cd shmip_E && python plot_shmip.py
Output: outputs/shmip_E.png
"""

import os
import sys

sys.argv = sys.argv[:1]

CHK_DIR  = "outputs/checkpoints"
OUTFILE  = "outputs/shmip_E.png"
CASES_ALL = ["E1", "E2", "E3", "E4", "E5"]

cases = [c for c in CASES_ALL
         if os.path.exists(os.path.join(CHK_DIR, f"{c}.h5"))]

if not cases:
    print(f"No checkpoints found in {CHK_DIR}/. "
          f"Run the simulation first (python run_shmip_E.py).")
    sys.exit(1)

checkpoints = [os.path.join(CHK_DIR, f"{c}.h5") for c in cases]

print(f"Suite E ribbon plot: {cases}")
print(f"  checkpoints: {CHK_DIR}/")
print(f"  output:      {OUTFILE}")

cases = cases[::-1]
checkpoints = checkpoints[::-1]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from shmip_summary_3d import make_ribbon_figure

make_ribbon_figure(
    checkpoints=checkpoints,
    labels=cases,
    outfile=OUTFILE,
    suite="E",
    band_width_E=200.0,
    print_diag=True,
)
