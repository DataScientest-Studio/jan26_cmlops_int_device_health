#!/usr/bin/env python3
"""
Create demo dataset for defense presentation.

Generates sample signals in data/samples/ that demonstrate:
  1. Healthy device signals (Gaussian, well-centered, low noise)
  2. Unhealthy device signals (Lorentzian, off-center, noisy)
  3. Data-drift signals (shifted center, high noise — sensor degradation)
  4. Concept-drift signals (boundary parameters — process change)

Each scenario has 5 individual JSON files + 1 batch CSV per scenario.
Files are compatible with the /predict API endpoint and the Streamlit
Predictions page file-upload feature.

Usage:
    python scripts/create_demo_dataset.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.signal_generator import generate_signal  # noqa: E402

SAMPLES_DIR = ROOT / "data" / "samples"

SCENARIOS: list[dict] = [
    {
        "name": "healthy",
        "description": "Normal Gaussian signals — well-centered, narrow, high amplitude, low noise",
        "seeds": [100, 101, 102, 103, 104],
        "params": {"shape_type": "gaussian", "drift_scenario": "baseline"},
    },
    {
        "name": "unhealthy",
        "description": "Lorentzian signals — off-center, wide, low amplitude, noisy",
        "seeds": [200, 201, 202, 203, 204],
        "params": {"shape_type": "lorentzian", "drift_scenario": "baseline"},
    },
    {
        "name": "data_drift",
        "description": "Sensor degradation — shifted center + high noise",
        "seeds": [300, 301, 302, 303, 304],
        "params": {"shape_type": "gaussian", "drift_scenario": "data_drift"},
    },
    {
        "name": "concept_drift",
        "description": "Process change — parameters shift toward boundary conditions",
        "seeds": [400, 401, 402, 403, 404],
        "params": {"shape_type": "gaussian", "drift_scenario": "concept_drift"},
    },
]


def _signal_to_json(sig) -> dict:
    """Convert a LabeledSignal to a JSON-serialisable /predict request body."""
    time_vals = list(sig.signal.time)
    amp_vals = [None if v is None or v != v else float(v) for v in sig.signal.amplitude]
    label_name = "healthy" if sig.label == 0 else "unhealthy"
    return {
        "device_id": "",
        "device_name": f"Demo-{label_name}",
        "device_type": "Sensor-Demo",
        "location": "Demo-Lab",
        "time_values": time_vals,
        "amplitude_values": amp_vals,
    }


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    total_files = 0

    for scenario in SCENARIOS:
        name = scenario["name"]
        desc = scenario["description"]
        seeds = scenario["seeds"]
        params = scenario["params"]
        scenario_dir = SAMPLES_DIR / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        all_payloads: list[dict] = []
        for i, seed in enumerate(seeds, start=1):
            sig = generate_signal(**params, seed=seed)
            payload = _signal_to_json(sig)
            payload["device_name"] = f"Demo-{name}-{i:03d}"

            # Individual JSON file
            json_path = scenario_dir / f"{name}_{i:03d}.json"
            json_path.write_text(json.dumps(payload, indent=2) + "\n")
            all_payloads.append(payload)
            total_files += 1

        # Batch CSV (columns: time_values, amplitude_values as semicolon-delimited)
        csv_path = scenario_dir / f"{name}_batch.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["device_name", "time_values", "amplitude_values"])
            for p in all_payloads:
                tv = ";".join(str(v) for v in p["time_values"])
                av = ";".join("" if v is None else str(v) for v in p["amplitude_values"])
                writer.writerow([p["device_name"], tv, av])
        total_files += 1

        print(f"  ✓ {name:15s}  ({desc})")

    # Write a README inside data/samples/
    readme = SAMPLES_DIR / "README.md"
    readme.write_text(
        "# Demo Dataset\n\n"
        "Pre-generated signals for the defense demo.\n\n"
        "| Scenario | Description |\n"
        "| --- | --- |\n"
        + "".join(f"| `{s['name']}/` | {s['description']} |\n" for s in SCENARIOS)
        + "\n"
        "## Usage\n\n"
        "**Single signal (JSON):**\n"
        "```bash\n"
        "curl -X POST http://localhost:80/predict \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d @data/samples/healthy/healthy_001.json\n"
        "```\n\n"
        "**UI upload:** Open Streamlit → 🎯 Predictions → Upload JSON file.\n"
    )
    total_files += 1

    print(f"\nCreated {total_files} files in {SAMPLES_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
