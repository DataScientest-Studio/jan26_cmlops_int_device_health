#!/usr/bin/env python3
"""Generate sample signal files for the Predictions page."""

from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.signal_processing.signal_generator import generate_signal

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
os.makedirs(OUT, exist_ok=True)

# ── Single healthy (Gaussian) signal ──────────────────────────
sig_h = generate_signal(
    shape_type="gaussian", mu=50.0, width_param=2.5, height=2.75, noise_level=0.015, seed=42
)
t_h = list(sig_h.signal.time)
a_h = list(sig_h.signal.amplitude)

with open(os.path.join(OUT, "healthy_signal.json"), "w") as f:
    json.dump({"time": t_h, "amplitude": a_h}, f, indent=2)

with open(os.path.join(OUT, "healthy_signal.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time", "amplitude"])
    for t, a in zip(t_h, a_h, strict=True):
        w.writerow([t, a])

# ── Single unhealthy (Lorentzian) signal ──────────────────────
sig_u = generate_signal(
    shape_type="lorentzian", mu=50.0, width_param=5.0, height=1.25, noise_level=0.08, seed=42
)
t_u = list(sig_u.signal.time)
a_u = list(sig_u.signal.amplitude)

with open(os.path.join(OUT, "unhealthy_signal.json"), "w") as f:
    json.dump({"time": t_u, "amplitude": a_u}, f, indent=2)

with open(os.path.join(OUT, "unhealthy_signal.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time", "amplitude"])
    for t, a in zip(t_u, a_u, strict=True):
        w.writerow([t, a])

# ── Batch JSON (3 healthy + 2 unhealthy) ─────────────────────
batch = []
for i in range(3):
    s = generate_signal(
        shape_type="gaussian",
        mu=50.0,
        width_param=2.5,
        height=2.75,
        noise_level=0.015,
        seed=100 + i,
    )
    batch.append(
        {
            "id": f"healthy-{i + 1}",
            "time": list(s.signal.time),
            "amplitude": list(s.signal.amplitude),
        }
    )

for i in range(2):
    s = generate_signal(
        shape_type="lorentzian",
        mu=50.0,
        width_param=5.0,
        height=1.25,
        noise_level=0.08,
        seed=200 + i,
    )
    batch.append(
        {
            "id": f"unhealthy-{i + 1}",
            "time": list(s.signal.time),
            "amplitude": list(s.signal.amplitude),
        }
    )

with open(os.path.join(OUT, "batch_signals.json"), "w") as f:
    json.dump(batch, f, indent=2)

# ── Batch CSV (wide format, semicolon-separated values) ──────
with open(os.path.join(OUT, "batch_signals.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["device_name", "time_values", "amplitude_values"])
    for sig in batch:
        t_str = ";".join(f"{v:.6f}" for v in sig["time"])
        a_str = ";".join(f"{v:.6f}" for v in sig["amplitude"])
        w.writerow([sig["id"], t_str, a_str])

print("Created sample files:")
for fn in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, fn))
    print(f"  data/samples/{fn}  ({size:,} bytes)")
