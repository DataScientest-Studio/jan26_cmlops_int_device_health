"""
Airflow DAG for manual model promotion and rollback.

This DAG can be manually triggered to:
1. Promote a specific model version to Production
2. Rollback to a previous Champion model
3. Archive old models (by retention period or explicit version)
4. Transition models between stages (Staging → Production → Archived)

Trigger params:
    - action: "promote" | "rollback" | "archive"
    - model_version: version number to promote/rollback to (required for promote/rollback)
    - force: bool to bypass safety checks
    - retention_days: archive models older than N days (archive action, period mode)
    - archive_model_version: explicit version to archive (archive action, version mode)

Schedule: Manual trigger only (no schedule)
"""

import contextlib
import os
from datetime import datetime, timedelta

from airflow.models import Param
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from airflow import DAG

try:
    from _dag_guards import require_cloud_mode
except ModuleNotFoundError:
    from airflow.dags._dag_guards import require_cloud_mode

_DAG_ID = "model_promotion"


def _get_model_name():  # type: ignore[return]
    """Return the registered model name, preferring MODEL_REGISTRY_NAME."""
    return os.getenv("MODEL_REGISTRY_NAME") or os.getenv("MODEL_NAME", "device_health_classifier")


def get_model_info(**context):  # type: ignore[return]
    """
    Get info about the current champion (or a specific version).
    Non-fatal: returns empty dict if no champion found (e.g. fresh registry).
    """
    require_cloud_mode(_DAG_ID)
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.tracking.MlflowClient()
    model_name = _get_model_name()

    print(f"Using model registry: {model_name}")

    params = context.get("params", {})
    model_version = params.get("model_version")

    if model_version is None:
        try:
            mv = client.get_model_version_by_alias(model_name, "champion")
        except Exception:
            print("No 'champion' alias found - registry may be empty (greenfield state)")
            return {"champion_version": None, "model_name": model_name}
    else:
        mv = client.get_model_version(model_name, str(model_version))

    run = client.get_run(mv.run_id)
    # Force JSON-safe types: mv.aliases is a protobuf RepeatedScalarContainer,
    # not a plain list, which causes XCom serialization failures.
    aliases = list(getattr(mv, "aliases", None) or [])

    info = {
        "version": str(mv.version),
        "aliases": aliases,
        "run_id": str(mv.run_id),
        "creation_timestamp": int(mv.creation_timestamp),
        "metrics": {k: float(v) for k, v in run.data.metrics.items()},
        "model_name": model_name,
        "champion_version": str(mv.version),
    }

    print(f"Model info: version={mv.version}, aliases={aliases}")
    return info


def branch_by_action(**context):  # type: ignore[return]
    """Route to the correct task based on the 'action' DAG parameter."""
    action = context["params"].get("action", "promote")
    if action == "promote":
        return "promote_model"
    elif action == "rollback":
        return "rollback_model"
    else:
        return "archive_models"


def promote_to_production(**context):  # type: ignore[return]
    """
    Promote a model version to Champion by setting the 'champion' alias.
    """
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.tracking.MlflowClient()
    model_name = _get_model_name()

    params = context["params"]
    model_version = params.get("model_version")
    force = params.get("force", False)

    if not model_version:
        raise ValueError("model_version parameter is required for promote action")

    print(f"Promoting model v{model_version} to Champion (force={force})")

    all_versions = client.search_model_versions(f"name='{model_name}'")
    current_champions = []
    for v in all_versions:
        v_aliases = getattr(v, "aliases", []) or []
        if "champion" in v_aliases:
            current_champions.append(v)

    # Validation: Check F1 unless force=True
    if not force and current_champions:
        current_champion = current_champions[0]
        current_run = client.get_run(current_champion.run_id)
        current_f1 = current_run.data.metrics.get(
            "test_f1_score", 0
        ) or current_run.data.metrics.get("test_f1", 0)

        new_model = client.get_model_version(model_name, str(model_version))
        new_run = client.get_run(new_model.run_id)
        new_f1 = new_run.data.metrics.get("test_f1_score", 0) or new_run.data.metrics.get(
            "test_f1", 0
        )

        if new_f1 <= current_f1:
            raise ValueError(
                f"New model F1 ({new_f1:.4f}) not better than current champion "
                f"({current_f1:.4f}). Use force=True to override."
            )

    # Remove old champion alias
    for champion in current_champions:
        with contextlib.suppress(Exception):
            client.delete_registered_model_alias(model_name, "champion")
        print(f"Removed champion alias from v{champion.version}")

    # Set champion alias on new model
    client.set_registered_model_alias(model_name, "champion", str(model_version))

    client.set_model_version_tag(
        name=model_name,
        version=str(model_version),
        key="promoted_at",
        value=datetime.now().isoformat(),
    )
    client.set_model_version_tag(
        name=model_name,
        version=str(model_version),
        key="promoted_by",
        value="airflow_model_promotion_dag",
    )

    result = {
        "action": "promote",
        "model_version": model_version,
        "model_name": model_name,
        "promoted_at": datetime.now().isoformat(),
        "previous_champion": current_champions[0].version if current_champions else None,
    }

    print(f"Promoted v{model_version} to Champion: {result}")
    return result


def rollback_to_version(**context):  # type: ignore[return]
    """Rollback champion to a previous model version."""
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.tracking.MlflowClient()
    model_name = _get_model_name()

    params = context["params"]
    model_version = params.get("model_version")

    if not model_version:
        raise ValueError("model_version parameter is required for rollback action")

    print(f"Rolling back champion to v{model_version}")

    prev_version = None
    try:
        current_champion = client.get_model_version_by_alias(model_name, "champion")
        prev_version = current_champion.version
    except Exception:
        print("No current champion alias found - setting directly")

    client.set_registered_model_alias(model_name, "champion", str(model_version))

    client.set_model_version_tag(
        name=model_name,
        version=str(model_version),
        key="rolled_back_at",
        value=datetime.now().isoformat(),
    )
    if prev_version:
        client.set_model_version_tag(
            name=model_name,
            version=str(model_version),
            key="rolled_back_from",
            value=str(prev_version),
        )

    result = {
        "action": "rollback",
        "model_name": model_name,
        "rolled_back_to": model_version,
        "rolled_back_from": prev_version,
        "rolled_back_at": datetime.now().isoformat(),
    }

    print(f"Rolled back to v{model_version}: {result}")
    return result


def archive_old_models(**context):  # type: ignore[return]
    """
    Archive model versions.

    Two modes:
      * archive_model_version is set -> archive that explicit version
      * otherwise use retention_days to archive unaliased versions older than N days
    """
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.tracking.MlflowClient()
    model_name = _get_model_name()

    params = context["params"]
    retention_days = params.get("retention_days", 90)
    explicit_version = params.get("archive_model_version")

    all_versions = client.search_model_versions(f"name='{model_name}'")
    archived_count = 0

    if explicit_version is not None:
        print(f"Archiving explicit version v{explicit_version}")
        target = next((v for v in all_versions if int(v.version) == int(explicit_version)), None)
        if target is None:
            raise ValueError(f"Model version {explicit_version} not found in {model_name}")

        # Safety: do NOT archive the current champion
        try:
            champ = client.get_model_version_by_alias(model_name, "champion")
            if int(champ.version) == int(explicit_version):
                raise ValueError(
                    f"Cannot archive v{explicit_version} - it is the current champion. "
                    "Promote another version first."
                )
        except ValueError:
            raise
        except Exception:
            pass

        with contextlib.suppress(Exception):
            client.set_model_version_tag(
                name=model_name,
                version=str(explicit_version),
                key="archived_by",
                value="airflow_model_promotion_dag",
            )
        with contextlib.suppress(Exception):
            client.set_model_version_tag(
                name=model_name,
                version=str(explicit_version),
                key="archived_at",
                value=datetime.now().isoformat(),
            )

        # Remove challenger alias if this version holds it
        try:
            chall = client.get_model_version_by_alias(model_name, "challenger")
            if int(chall.version) == int(explicit_version):
                with contextlib.suppress(Exception):
                    client.delete_registered_model_alias(model_name, "challenger")
                print(f"Removed challenger alias from v{explicit_version}")
        except Exception:
            pass

        archived_count = 1
        print(f"Archived v{explicit_version}")

    else:
        print(f"Archiving models older than {retention_days} days")

        cutoff_ms = (datetime.now() - timedelta(days=retention_days)).timestamp() * 1000

        # Protect champion + challenger from retention-based archiving
        protected_versions = set()
        for alias in ("champion", "challenger"):
            try:
                mv = client.get_model_version_by_alias(model_name, alias)
                protected_versions.add(mv.version)
            except Exception:
                pass

        for version in all_versions:
            if version.version in protected_versions:
                continue
            if version.creation_timestamp < cutoff_ms:
                with contextlib.suppress(Exception):
                    client.set_model_version_tag(
                        name=model_name,
                        version=version.version,
                        key="archived_by",
                        value="airflow_model_promotion_dag",
                    )
                with contextlib.suppress(Exception):
                    client.set_model_version_tag(
                        name=model_name,
                        version=version.version,
                        key="archived_at",
                        value=datetime.now().isoformat(),
                    )
                archived_count += 1
                print(f"Archived v{version.version}")

    result = {
        "action": "archive",
        "model_name": model_name,
        "archived_count": archived_count,
        "mode": "explicit" if explicit_version is not None else "retention",
        "retention_days": retention_days if explicit_version is None else None,
        "archived_version": explicit_version,
    }

    print(f"Archive complete: {result}")
    return result


def validate_production_model(**context):  # type: ignore[return]
    """
    Smoke-test the current champion model with synthetic 6-feature input.
    Skipped gracefully if champion does not exist.
    """
    import mlflow
    import numpy as np

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_name = _get_model_name()

    print("Running smoke test on champion model...")

    # 6 features produced by the signal feature extractor:
    # amplitude_mean, amplitude_std, peak_value, signal_duration, snr_estimate, kurtosis
    N_FEATURES = 6

    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
    except Exception as exc:
        print(f"Could not load champion model - skipping smoke test: {exc}")
        return

    np.random.seed(42)
    test_features = np.random.rand(5, N_FEATURES)

    import pandas as pd

    predictions = model.predict(pd.DataFrame(test_features))
    assert len(predictions) == 5, f"Expected 5 predictions, got {len(predictions)}"

    print(f"Smoke test passed: {len(predictions)} predictions validated")


def send_promotion_notification(**context):  # type: ignore[return]
    """Log a summary notification for the completed action."""
    task_instance = context["task_instance"]
    params = context["params"]

    action = params.get("action", "unknown")

    result = None
    for task_id in ("promote_model", "rollback_model", "archive_models"):
        result = task_instance.xcom_pull(task_ids=task_id)
        if result is not None:
            break

    message = (
        f"\nModel Promotion DAG - {action.upper()} Complete\n"
        f"Timestamp : {datetime.now().isoformat()}\n"
        f"Model     : {_get_model_name()}\n"
        f"Details   : {result}\n"
    )
    print(message)


# ============================================================================
# DAG Definition
# ============================================================================

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="model_promotion",
    description="Manual model promotion, rollback, and lifecycle management",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=["mlflow", "promotion", "manual"],
    max_active_runs=1,
    params={
        "action": Param(
            "promote",
            type="string",
            enum=["promote", "rollback", "archive"],
            description="Action to perform",
        ),
        "model_version": Param(
            None,
            type=["null", "integer"],
            description="Version to promote/rollback to (required for those actions)",
        ),
        "force": Param(
            False,
            type="boolean",
            description="Bypass F1 improvement check when promoting",
        ),
        "retention_days": Param(
            90,
            type="integer",
            description="Archive versions older than N days (archive action, period mode)",
        ),
        "archive_model_version": Param(
            None,
            type=["null", "integer"],
            description="Explicit version to archive (overrides retention_days when set)",
        ),
    },
) as dag:
    get_info_task = PythonOperator(
        task_id="get_current_info",
        python_callable=get_model_info,
        provide_context=True,
    )

    branch_task = BranchPythonOperator(
        task_id="branch_action",
        python_callable=branch_by_action,
        provide_context=True,
    )

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_to_production,
        provide_context=True,
    )

    rollback_task = PythonOperator(
        task_id="rollback_model",
        python_callable=rollback_to_version,
        provide_context=True,
    )

    archive_task = PythonOperator(
        task_id="archive_models",
        python_callable=archive_old_models,
        provide_context=True,
    )

    validate_task = PythonOperator(
        task_id="validate_production",
        python_callable=validate_production_model,
        provide_context=True,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    notify_task = PythonOperator(
        task_id="send_notification",
        python_callable=send_promotion_notification,
        provide_context=True,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Dependency graph: branch after info, then validate+notify regardless of action
    get_info_task >> branch_task >> [promote_task, rollback_task, archive_task]
    promote_task >> validate_task
    rollback_task >> validate_task
    archive_task >> validate_task
    validate_task >> notify_task
