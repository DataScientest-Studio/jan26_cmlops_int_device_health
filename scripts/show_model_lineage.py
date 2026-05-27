#!/usr/bin/env python
"""
UC-11: Model Lineage & Reproducibility.

Queries the MLflow tracking server (local or DagsHub) and Git/DVC to
display a full audit trail: experiment runs, metrics, parameters, the
Git commit hash, and the DVC data version used for each run.

Usage:
    python scripts/show_model_lineage.py
    python scripts/show_model_lineage.py --experiment mlops_device_health
    python scripts/show_model_lineage.py --tracking-uri http://localhost:5000
    python scripts/show_model_lineage.py --run-id <mlflow-run-id>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _git_log(n: int = 5) -> list[dict]:
    """Return recent git commits."""
    try:
        result = subprocess.run(
            ["git", "--no-pager", "log", f"-{n}", "--format=%H|%s|%cr|%an"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        commits = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                commits.append(
                    {
                        "hash": parts[0][:12],
                        "message": parts[1],
                        "when": parts[2],
                        "author": parts[3],
                    }
                )
        return commits
    except Exception:
        return []


def _dvc_status() -> str:
    """Return DVC remote status summary."""
    try:
        result = subprocess.run(
            ["dvc", "status", "--cloud"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip() or "DVC remote is in sync."
    except FileNotFoundError:
        return "dvc not found — install dvc to track data versions."
    except subprocess.TimeoutExpired:
        return "DVC status timed out (check DagsHub connectivity)."
    except Exception as exc:
        return str(exc)


def _list_mlflow_runs(tracking_uri: str, experiment: str, limit: int = 10) -> list[dict]:
    """List recent MLflow runs."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            return []

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=limit,
            order_by=["start_time DESC"],
        )
        result = []
        for run in runs:
            metrics = run.data.metrics
            params = run.data.params
            tags = run.data.tags
            result.append(
                {
                    "run_id": run.info.run_id[:12],
                    "status": run.info.status,
                    "accuracy": metrics.get("accuracy", metrics.get("test_accuracy", "—")),
                    "f1": metrics.get("f1_score", metrics.get("f1", "—")),
                    "model": params.get("model_type", params.get("model", "—")),
                    "git_commit": tags.get("mlflow.source.git.commit", "—")[:12],
                    "note": tags.get("mlflow.note.content", ""),
                }
            )
        return result
    except Exception as exc:
        return [{"error": str(exc)}]


def main() -> int:
    parser = argparse.ArgumentParser(description="UC-11: Model Lineage & Reproducibility")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"),
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or http://localhost:5001).",
    )
    parser.add_argument("--experiment", default="mlops_device_health", help="Experiment name.")
    parser.add_argument("--limit", type=int, default=5, help="Max runs to display.")
    parser.add_argument("--run-id", default=None, help="Show details for a specific run ID.")
    args = parser.parse_args()

    print("UC-11: Model Lineage & Reproducibility")
    print("=" * 60)

    # ── Git history ───────────────────────────────────────────────────────────
    print("\n📌  Recent Git commits:")
    commits = _git_log(5)
    if commits:
        for c in commits:
            print(f"  {c['hash']}  {c['when']:>12}  {c['message'][:60]}")
    else:
        print("  (git not available or no commits)")

    # ── DVC status ────────────────────────────────────────────────────────────
    print("\n🗂️   DVC data version status:")
    print(f"  {_dvc_status()}")

    # ── MLflow runs ───────────────────────────────────────────────────────────
    print(f"\n🧪  MLflow runs (experiment: {args.experiment}):")
    runs = _list_mlflow_runs(args.tracking_uri, args.experiment, args.limit)
    if not runs:
        print(f"  No runs found in experiment '{args.experiment}'.")
        print(f"  Is MLflow running at {args.tracking_uri}?")
        print("  (docker compose up -d mlflow)")
    elif runs and "error" in runs[0]:
        print(f"  MLflow not reachable: {runs[0]['error']}")
        print("  Start MLflow: docker compose up -d mlflow")
    else:
        header = f"  {'Run ID':>12}  {'Status':>10}  {'Accuracy':>8}  {'F1':>6}  {'Git':>12}  Model"
        print(header)
        print("  " + "-" * 62)
        for r in runs:
            acc = f"{r['accuracy']:.4f}" if isinstance(r["accuracy"], float) else r["accuracy"]
            f1 = f"{r['f1']:.4f}" if isinstance(r["f1"], float) else r["f1"]
            print(
                f"  {r['run_id']:>12}  {r['status']:>10}  {acc:>8}  {f1:>6}  "
                f"{r['git_commit']:>12}  {r['model']}"
            )

    # ── Model registry ────────────────────────────────────────────────────────
    print("\n🏆  Model Registry (Production / Staging):")
    try:
        import mlflow

        mlflow.set_tracking_uri(args.tracking_uri)
        client = mlflow.tracking.MlflowClient()
        _model_name = os.environ.get("MODEL_REGISTRY_NAME", "device_health_classifier")
        for stage in ("Production", "Staging"):
            all_versions = client.search_model_versions(f"name='{_model_name}'")
            stage_versions = sorted(
                [v for v in all_versions if v.current_stage == stage],
                key=lambda v: int(v.version),
                reverse=True,
            )[:1]
            if stage_versions:
                v = stage_versions[0]
                print(
                    f"  {stage}: v{v.version}  run_id={(v.run_id or '')[:12]}  "
                    f"source={str(v.source)[:50] if v.source else '\u2014'}"
                )
            else:
                print(f"  {stage}: —")
    except Exception as exc:
        print(f"  (Registry unavailable: {exc})")

    print("\n✅  Lineage report complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
