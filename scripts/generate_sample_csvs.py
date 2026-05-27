#!/usr/bin/env python3
"""Generate example CSV signal files for predictions testing.

Signal format requirements:
  - time must span [0, 100]  (signal_models.py: "Time array must span [0, 100]")
  - at least 51 data points
  - columns: time, amplitude
"""

import csv
import math
import random

import numpy as np

N = 200
# Time array must span [0, 100] exactly
time_array = list(np.linspace(0, 100, N))

# ── Healthy signal: low-amplitude near-sinusoidal vibration ──────────────────
rows_healthy = []
for i, t in enumerate(time_array):
    amp = (
        0.15 * math.sin(2 * math.pi * 0.5 * t)  # 0.5 Hz fundamental
        + 0.05 * math.sin(2 * math.pi * 1.0 * t)  # 1 Hz 2nd harmonic
        + 0.02 * (0.5 - (i % 7) / 7)  # tiny low-freq wander
    )
    rows_healthy.append({"time": round(t, 6), "amplitude": round(amp, 6)})

with open("data/samples/healthy_signal.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["time", "amplitude"])
    writer.writeheader()
    writer.writerows(rows_healthy)

print("Created data/samples/healthy_signal.csv   (200 rows, time 0–100, healthy device signal)")

# ── Unhealthy signal: high amplitude, noisy, with spike transient ────────────
random.seed(0)
rows_unhealthy = []
for i, t in enumerate(time_array):
    noise = random.gauss(0, 0.3)
    spike = 2.5 if 90 <= i <= 95 else 0.0  # sharp transient impact event
    amp = (
        0.9 * math.sin(2 * math.pi * 0.5 * t)
        + 0.4 * math.sin(2 * math.pi * 0.73 * t)  # sub-harmonic interference
        + noise
        + spike
    )
    rows_unhealthy.append({"time": round(t, 6), "amplitude": round(amp, 6)})

with open("data/samples/unhealthy_signal.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["time", "amplitude"])
    writer.writeheader()
    writer.writerows(rows_unhealthy)

print(
    "Created data/samples/unhealthy_signal.csv (200 rows, time 0–100, unhealthy signal with spike + noise)"
)
