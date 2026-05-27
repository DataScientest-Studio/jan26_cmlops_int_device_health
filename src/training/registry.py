"""
MLflow Model Registry Management.

This module provides functions for managing model lifecycle in MLflow Model Registry:
- Register models from training runs
- Promote models between stages (None → Staging → Production)
- Archive old models
- Query production and staging models

Model Stages:
- None: Newly registered models (default)
- Staging: Models being validated/tested
- Production: Active production model
- Archived: Deprecated models

Usage:
    >>> from src.training.registry import register_model, promote_model
    >>>
    >>> # Register a model from a training run
    >>> version = register_model(run_id="abc123", model_name="device_health_classifier")
    >>>
    >>> # Promote to staging for evaluation
    >>> promote_model("device_health_classifier", version=1, stage="Staging")
    >>>
    >>> # After validation, promote to production
    >>> promote_model("device_health_classifier", version=1, stage="Production")
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def register_model(
    run_id: str,
    model_name: str,
    description: str | None = None,
) -> int:
    """
    Register a model from an MLflow run to the Model Registry.

    Args:
        run_id: MLflow run ID containing the model
        model_name: Name for the registered model
        description: Optional description of this model version

    Returns:
        Model version number (integer)

    Raises:
        ValueError: If run doesn't exist or model artifact not found

    Example:
        >>> version = register_model(
        ...     run_id="abc123def456",
        ...     model_name="device_health_classifier",
        ...     description="Logistic Regression with C=1.0"
        ... )
        >>> print(f"Registered as version {version}")
    """
    client = MlflowClient()

    # Verify run exists
    try:
        run = client.get_run(run_id)
    except Exception as e:
        raise ValueError(f"Run {run_id} not found: {e}") from e

    # Build model URI
    model_uri = f"runs:/{run_id}/model"

    # Register model (creates model if doesn't exist)
    try:
        # Ensure the registered model exists (idempotent)
        import contextlib

        with contextlib.suppress(Exception):
            client.create_registered_model(model_name)

        # MLflow v3: use create_model_version directly (mlflow.register_model
        # calls search_logged_models which may fail on older servers)
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
        )

        # Update description if provided
        if description:
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description,
            )

        # Add tags from the run for traceability
        tags_to_copy = ["model_version", "algorithm", "git_commit", "dvc_data_version"]
        for tag_key in tags_to_copy:
            if tag_key in run.data.tags:
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=tag_key,
                    value=run.data.tags[tag_key],
                )

        return int(model_version.version)

    except Exception as e:
        raise ValueError(f"Failed to register model: {e}") from e


def promote_model(
    model_name: str,
    version: int,
    stage: str,
    archive_existing_production: bool = False,
) -> None:
    """
    Promote a model to a specific stage in the Model Registry.

    Uses both the legacy stage API (for servers that still support it) and
    the modern alias API (MLflow ≥ 2.9 / DagsHub) so the promotion is
    discoverable regardless of which query path is used.

    Aliases set:
        "Production" → alias ``champion``
        "Staging"    → alias ``challenger``

    Args:
        model_name: Registered model name
        version: Model version number
        stage: Target stage ("Staging", "Production", "Archived", or "None")
        archive_existing_production: If True, archive current production models
            when promoting to Production

    Raises:
        ValueError: If model/version doesn't exist or invalid stage
    """
    valid_stages = ["Staging", "Production", "Archived", "None"]
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of: {valid_stages}")

    client = MlflowClient()

    # Verify model version exists
    try:
        client.get_model_version(model_name, str(version))
    except Exception as e:
        raise ValueError(f"Model {model_name} version {version} not found: {e}") from e

    # Archive existing production models if requested
    if stage == "Production" and archive_existing_production:
        production_models = get_production_models(model_name)
        for prod_model in production_models:
            if prod_model["version"] != version:
                # Clear any alias pointing to the old champion
                for _alias in ("champion", "challenger"):
                    try:
                        mv = client.get_model_version_by_alias(model_name, _alias)
                        if str(mv.version) == str(prod_model["version"]):
                            client.delete_registered_model_alias(model_name, _alias)
                    except Exception:
                        pass

    # ── Modern alias API (primary path for MLflow 3.x) ──
    stage_to_alias = {
        "Production": "champion",
        "Staging": "challenger",
    }
    if stage in ("Archived", "None"):
        # Clear any stale alias that previously pointed to this version so
        # that alias-based lookups no longer surface archived models.
        for _alias in stage_to_alias.values():
            try:
                mv = client.get_model_version_by_alias(model_name, _alias)
                if str(mv.version) == str(version):
                    client.delete_registered_model_alias(model_name, _alias)
            except Exception:
                pass  # alias doesn't exist or server doesn't support it
    else:
        alias = stage_to_alias.get(stage)
        if alias:
            try:
                client.set_registered_model_alias(model_name, alias, str(version))
            except Exception as exc:
                # Only warn — the legacy transition above may have succeeded
                print(f"⚠ Failed to set alias '{alias}' on {model_name} v{version}: {exc}")


def archive_model(model_name: str, version: int) -> None:
    """
    Archive a model version (shorthand for promote to Archived stage).

    Args:
        model_name: Registered model name
        version: Model version number

    Example:
        >>> archive_model("device_health_classifier", version=1)
    """
    promote_model(model_name, version, stage="Archived")


def get_production_models(model_name: str) -> list[dict[str, Any]]:
    """
    Get all production models for a given model name.

    Args:
        model_name: Registered model name

    Returns:
        List of dicts with model version info:
        [
            {
                "version": int,
                "run_id": str,
                "stage": "Production",
                "metrics": dict,
                "params": dict,
                "tags": dict,
            },
            ...
        ]

    Example:
        >>> production_models = get_production_models("device_health_classifier")
        >>> if production_models:
        ...     champion = production_models[0]
        ...     print(f"Champion version: {champion['version']}")
        ...     print(f"Accuracy: {champion['metrics']['test_accuracy']:.2%}")
    """
    return _get_models_by_stage(model_name, "Production")


def get_staging_models(model_name: str) -> list[dict[str, Any]]:
    """
    Get all staging models for a given model name.

    Args:
        model_name: Registered model name

    Returns:
        List of dicts with model version info (same format as get_production_models)

    Example:
        >>> staging_models = get_staging_models("device_health_classifier")
        >>> for model in staging_models:
        ...     print(f"Challenger v{model['version']}: {model['metrics']['test_accuracy']:.2%}")
    """
    return _get_models_by_stage(model_name, "Staging")


def get_latest_model_version(model_name: str) -> int | None:
    """
    Get the latest version number for a registered model.

    Args:
        model_name: Registered model name

    Returns:
        Latest version number, or None if model doesn't exist

    Example:
        >>> latest = get_latest_model_version("device_health_classifier")
        >>> print(f"Latest version: {latest}")
    """
    client = MlflowClient()

    try:
        client.get_registered_model(model_name)
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            return max(int(v.version) for v in versions)
        return None
    except Exception:
        return None


def list_registered_models() -> list[dict[str, Any]]:
    """
    List all registered models in the Model Registry.

    Returns:
        List of dicts with model info:
        [
            {
                "name": str,
                "creation_time": int,
                "last_updated": int,
                "latest_version": int,
                "production_versions": list[int],
                "staging_versions": list[int],
            },
            ...
        ]

    Example:
        >>> models = list_registered_models()
        >>> for model in models:
        ...     print(f"{model['name']}: v{model['latest_version']} "
        ...           f"(prod: {model['production_versions']})")
    """
    client = MlflowClient()

    models_info = []
    registered_models = client.search_registered_models()

    for rm in registered_models:
        versions = client.search_model_versions(f"name='{rm.name}'")

        production_versions = [
            int(v.version) for v in versions if "champion" in (getattr(v, "aliases", None) or [])
        ]
        staging_versions = [
            int(v.version) for v in versions if "challenger" in (getattr(v, "aliases", None) or [])
        ]
        latest_version = max(int(v.version) for v in versions) if versions else 0

        models_info.append(
            {
                "name": rm.name,
                "creation_time": rm.creation_timestamp,
                "last_updated": rm.last_updated_timestamp,
                "latest_version": latest_version,
                "production_versions": production_versions,
                "staging_versions": staging_versions,
            }
        )

    return models_info


def load_production_model(model_name: str) -> Any:
    """
    Load the current production model artifact.

    Args:
        model_name: Registered model name

    Returns:
        Loaded sklearn model

    Raises:
        ValueError: If no production model exists

    Example:
        >>> model = load_production_model("device_health_classifier")
        >>> predictions = model.predict(X_test)
    """
    production_models = get_production_models(model_name)

    if not production_models:
        raise ValueError(f"No production model found for '{model_name}'")

    # Use the latest production version if multiple exist
    latest_prod = max(production_models, key=lambda m: m["version"])

    model_uri = f"models:/{model_name}/{latest_prod['version']}"
    model = mlflow.sklearn.load_model(model_uri)

    return model


def load_production_model_artifact(
    model_name: str, stage: str = "Production"
) -> dict[str, Any] | None:
    """
    Load the complete production model artifact from MLflow Registry.

    This function loads the full model artifact including scaler, feature names,
    and metadata, compatible with the training pipeline's model format.

    Args:
        model_name: Registered model name
        stage: Model stage to load ("Production" or "Staging"), default "Production"

    Returns:
        Model artifact dict with:
        {
            "model": trained classifier,
            "scaler": StandardScaler (if available),
            "semi_trainer": SemiSupervisedTrainer (if available),
            "feature_names": list[str],
            "model_version": str,
            "algorithm": str,
            "trained_at": str,
            "registry_version": int,
            "registry_stage": str,
            "mlflow_run_id": str,
        }
        Returns None if no model found in specified stage.

    Example:
        >>> artifact = load_production_model_artifact("device_health_classifier")
        >>> if artifact:
        ...     model = artifact["model"]
        ...     scaler = artifact["scaler"]
        ...     predictions = model.predict(scaler.transform(X))
        ... else:
        ...     print("No production model available, using bootstrap")
    """
    import pickle
    import tempfile

    client = MlflowClient()

    # Get models in specified stage
    if stage == "Production":
        stage_models = get_production_models(model_name)
    elif stage == "Staging":
        stage_models = get_staging_models(model_name)
    else:
        raise ValueError(f"Invalid stage '{stage}'. Must be 'Production' or 'Staging'")

    if not stage_models:
        return None

    # Use the latest version in the stage
    latest_model = max(stage_models, key=lambda m: m["version"])
    version = latest_model["version"]

    try:
        # Try to download the full model artifact (pickle file)
        # The training pipeline saves a pickle file with all components
        # NOTE: client.get_run() is intentionally omitted here — run metadata
        # is not needed for artifact download, and the extra HTTP call is slow
        # on corporate TLS-inspection proxies.
        artifacts = client.list_artifacts(latest_model["run_id"])

        # Look for model artifact pickle file.
        # The artifact may have any name — train.py saves to model_output_path
        # which may be "models/challenger" (no .pkl extension) or
        # "models/trained_model.pkl".  We search the root-level artifacts
        # for any non-directory file that can be unpickled as our dict format.
        _skip_extensions = {".json", ".yaml", ".yml", ".txt", ".csv", ".html", ".py"}
        _skip_names = {"conda.yaml", "python_env.yaml", "MLmodel", "requirements.txt"}
        model_artifact_path = None
        for artifact in artifacts:
            if artifact.is_dir:
                continue
            _name = artifact.path.rsplit("/", 1)[-1]
            if _name in _skip_names:
                continue
            _ext = "." + _name.rsplit(".", 1)[-1] if "." in _name else ""
            if _ext in _skip_extensions:
                continue
            # Candidate: ends in .pkl OR has no extension (e.g. "challenger")
            if _name.endswith(".pkl") or "." not in _name:
                model_artifact_path = artifact.path
                break

        if model_artifact_path:
            # Download the pickle file
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = client.download_artifacts(
                    latest_model["run_id"], model_artifact_path, tmpdir
                )
                try:
                    with open(local_path, "rb") as f:
                        model_artifact = pickle.load(f)  # noqa: S301
                except Exception:
                    model_artifact = None

                if isinstance(model_artifact, dict) and "model" in model_artifact:
                    # Add registry metadata — override model_version so the API
                    # reports the registry version, not the hardcoded training tag.
                    model_artifact["registry_version"] = version
                    model_artifact["registry_stage"] = stage
                    model_artifact["mlflow_run_id"] = latest_model["run_id"]
                    model_artifact["model_version"] = f"v{version}"
                    return model_artifact
                # File exists but could not be unpickled or has wrong format;
                # fall through to the sklearn fallback below.
                model_artifact_path = None
        model_uri = f"models:/{model_name}/{version}"
        sklearn_model = mlflow.sklearn.load_model(model_uri)

        # If the model is a Pipeline (e.g. StandardScaler → Classifier),
        # extract the scaler and the final estimator so that predict()
        # receives the same artifact shape as a bootstrap pickle.
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        extracted_scaler = None
        raw_model = sklearn_model
        if isinstance(sklearn_model, SklearnPipeline):
            for _step_name, step_obj in sklearn_model.steps:
                if isinstance(step_obj, StandardScaler):
                    extracted_scaler = step_obj
            # The final step is the actual classifier/regressor
            raw_model = sklearn_model.steps[-1][1]

        # Recover feature names from run params (logged as comma-separated)
        extracted_features: list[str] | None = None
        try:
            run = client.get_run(latest_model["run_id"])
            features_param = run.data.params.get("features", "")
            if features_param:
                extracted_features = [f.strip() for f in features_param.split(",") if f.strip()]
        except Exception:  # noqa: BLE001
            pass

        # Last resort: try model_metadata.json artifact
        if not extracted_features:
            try:
                import json as _json

                with tempfile.TemporaryDirectory() as tmpdir:
                    meta_path = client.download_artifacts(
                        latest_model["run_id"], "model_metadata.json", tmpdir
                    )
                    with open(meta_path) as fmeta:
                        meta = _json.load(fmeta)
                    extracted_features = meta.get("feature_names")
            except Exception:  # noqa: BLE001
                pass

        model_artifact = {
            "model": raw_model,
            "scaler": extracted_scaler,
            "feature_names": extracted_features,
            "model_version": f"registry_v{version}",
            "algorithm": latest_model["tags"].get("algorithm", "unknown"),
            "trained_at": latest_model["tags"].get("trained_at", "unknown"),
            "registry_version": version,
            "registry_stage": stage,
            "mlflow_run_id": latest_model["run_id"],
        }

        # If scaler is None, the pkl artifact was not uploaded (silent failure in
        # train.py mlflow.log_artifact) and the sklearn model is not a Pipeline.
        # Without a scaler, predict.py cannot scale features and will raise 400.
        # Return None here so the caller (get_model in dependencies.py) falls back
        # to the bootstrap pkl which always has a valid fitted scaler.
        if extracted_scaler is None:
            print(
                f"[WARN] Registry model v{version} has no scaler (pkl artifact missing or not a "
                f"Pipeline). Returning None to trigger bootstrap fallback."
            )
            return None

        return model_artifact

    except Exception as e:
        # Log the error but don't crash
        print(f"Warning: Failed to load model {model_name} v{version}: {e}")
        return None


def get_model_version_info(model_name: str, version: int) -> dict[str, Any]:
    """
    Get detailed information about a specific model version.

    Args:
        model_name: Registered model name
        version: Model version number

    Returns:
        Dict with detailed model version info:
        {
            "version": int,
            "stage": str,
            "run_id": str,
            "metrics": dict,
            "params": dict,
            "tags": dict,
            "creation_time": int,
            "description": str,
        }

    Example:
        >>> info = get_model_version_info("device_health_classifier", version=2)
        >>> print(f"Stage: {info['stage']}")
        >>> print(f"Accuracy: {info['metrics']['test_accuracy']:.2%}")
    """
    client = MlflowClient()

    try:
        model_version = client.get_model_version(model_name, str(version))
    except Exception as e:
        raise ValueError(f"Model {model_name} version {version} not found: {e}") from e

    # Get run info for metrics and params
    run = client.get_run(model_version.run_id)  # type: ignore[arg-type]

    # Derive effective stage from aliases (MLflow 3.x — current_stage is empty)
    aliases = getattr(model_version, "aliases", None) or []
    if "champion" in aliases:
        effective_stage = "Production"
    elif "challenger" in aliases:
        effective_stage = "Staging"
    else:
        effective_stage = "None"

    return {
        "version": int(model_version.version),
        "stage": effective_stage,
        "run_id": model_version.run_id,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
        "creation_time": model_version.creation_timestamp,
        "description": model_version.description or "",
    }


def _get_models_by_stage(model_name: str, stage: str) -> list[dict[str, Any]]:
    """
    Internal helper to get models by stage.

    Uses a two-pronged strategy for compatibility with both legacy MLflow
    stage API and modern alias-based API (MLflow ≥ 2.9 / DagsHub):

    1. **Alias lookup** — try ``get_model_version_by_alias`` with the alias
       that corresponds to the requested stage (Production → ``champion``,
       Staging → ``challenger``).  This is the preferred path on DagsHub and
       newer MLflow servers.
    2. **Legacy stage scan** — iterate ``search_model_versions`` results and
       check ``current_stage``.  Works on older MLflow servers.

    Results from both paths are merged (deduplicated by version number).

    Args:
        model_name: Registered model name
        stage: Model stage to filter by ("Production" or "Staging")

    Returns:
        List of dicts with model version info
    """
    stage_to_alias: dict[str, str] = {
        "Production": "champion",
        "Staging": "challenger",
    }

    client = MlflowClient()
    seen_versions: set[int] = set()
    stage_models: list[dict[str, Any]] = []

    def _version_to_dict(version_obj: Any, run_id: str | None = None) -> dict[str, Any] | None:
        """Convert a ModelVersion + its run into the standard dict.

        ``fetch_run=False`` path: build the dict from ModelVersion metadata only,
        skipping the extra ``get_run`` HTTP call.  Used for the alias-lookup path
        where run metrics/params are not needed for model loading.
        ``fetch_run=True`` (default) fetches full run data — needed for strategy 2.
        """
        rid = run_id or getattr(version_obj, "run_id", None)
        if not rid:
            return None
        return {
            "version": int(version_obj.version),
            "run_id": rid,
            "stage": stage,
            "metrics": {},
            "params": {},
            "tags": {},
        }

    def _version_to_dict_full(version_obj: Any, run_id: str | None = None) -> dict[str, Any] | None:
        """Convert a ModelVersion + its run into the standard dict (with run data)."""
        rid = run_id or getattr(version_obj, "run_id", None)
        if not rid:
            return None
        try:
            run = client.get_run(rid)
            return {
                "version": int(version_obj.version),
                "run_id": rid,
                "stage": stage,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "tags": dict(run.data.tags),
            }
        except Exception:
            return None

    # ── Strategy 1: alias lookup (MLflow ≥ 2.9 / DagsHub) ──
    # Preferred path.  Builds the entry without a ``get_run`` HTTP call — only
    # the alias endpoint is needed.  If alias lookup succeeds, skip strategy 2
    # entirely to avoid the expensive ``search_model_versions`` call.
    alias = stage_to_alias.get(stage)
    if alias:
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            # Use the full dict (with run metrics) so evaluate_promotion gets
            # real champion metrics instead of an empty dict.
            # The extra get_run call is acceptable for promotion decisions.
            entry = _version_to_dict_full(mv)
            if entry and entry["version"] not in seen_versions:
                stage_models.append(entry)
                seen_versions.add(entry["version"])
        except Exception:
            pass  # alias not set or server doesn't support aliases — fall through

    # Short-circuit: if alias lookup already found the model, skip the slow
    # ``search_model_versions`` scan (saves one extra HTTP round-trip).
    if stage_models:
        return stage_models

    # ── Strategy 2: legacy stage scan ──
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:
        print(f"\u26a0 search_model_versions failed for '{model_name}': {exc}")
        return stage_models  # return whatever the alias lookup found

    for version in versions:
        # Check if any alias matches the stage alias (MLflow 3.x primary path)
        v_aliases = getattr(version, "aliases", []) or []
        has_alias = alias is not None and alias in v_aliases

        # Also accept the legacy current_stage field as a fallback so the API
        # remains functional when aliases haven't been set (e.g. after manual
        # stage transitions via the old API, or after alias cleanup).
        has_legacy_stage = getattr(version, "current_stage", None) == stage

        if (has_alias or has_legacy_stage) and int(version.version) not in seen_versions:
            entry = _version_to_dict(version)
            if entry:
                stage_models.append(entry)
                seen_versions.add(entry["version"])

    return stage_models
