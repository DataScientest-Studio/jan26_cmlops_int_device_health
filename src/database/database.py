"""
Database helper class for MLOps device health monitoring.

Provides CRUD operations for:
- Storing predictions with signals and features
- Injecting sparse labels
- Querying realized accuracy
- Device management

Thread-safe for concurrent access.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from .init_db import (
    _migrate_drift_signals_raw_columns_sqlite,
    _migrate_model_approvals_challenger_test_sqlite,
    _migrate_model_training_data_sqlite,
    create_tables,
    create_tables_postgres,
    generate_device_id,
)

# ---------------------------------------------------------------------------
# PostgreSQL adapter — wraps psycopg2 so the rest of Database can use
# SQLite-style '?' placeholders and dict-row access unchanged.
# ---------------------------------------------------------------------------

# Track which PostgreSQL URLs have already had their schema bootstrapped in
# this process so we don't open a second DDL connection on every Database()
# instantiation.  The DDL is idempotent (CREATE TABLE IF NOT EXISTS), so
# running it once per URL per process is sufficient.
_pg_bootstrapped_urls: set[str] = set()


class _PgCursorWrapper:
    """
    Minimal psycopg2 cursor wrapper.

    - Translates '?' placeholders → '%s' at execute time.
    - Delegates row access to RealDictCursor so callers use row["col"]
      and dict(row) exactly as with sqlite3.Row.
    """

    __slots__ = ("_c",)

    def __init__(self, cursor) -> None:  # type: ignore[no-untyped-def]
        self._c = cursor

    def execute(self, sql: str, params: Any = ()) -> None:
        self._c.execute(sql.replace("?", "%s"), params if params else ())

    def executemany(self, sql: str, seq: Any) -> None:
        self._c.executemany(sql.replace("?", "%s"), seq)

    def fetchone(self) -> Any:
        return self._c.fetchone()

    def fetchall(self) -> list[Any]:
        return self._c.fetchall()

    @property
    def description(self) -> Any:
        """Expose cursor.description so callers can read column names."""
        return self._c.description

    @property
    def rowcount(self) -> int:
        return self._c.rowcount  # type: ignore[return-value]

    @property
    def lastrowid(self) -> Any:
        return self._c.lastrowid


class _PgConnectionWrapper:
    """
    Minimal psycopg2 connection wrapper.

    - cursor() returns _PgCursorWrapper, making every downstream
      cursor.execute() transparently handle '?' → '%s' conversion.
    - Skips SQLite-specific PRAGMA statements silently.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def cursor(self) -> _PgCursorWrapper:
        return _PgCursorWrapper(self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor))

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def execute(self, sql: str, params: Any = ()) -> None:
        """Support conn.execute() used for PRAGMA — silently skip for PostgreSQL."""
        if "PRAGMA" not in sql.upper():
            with self._conn.cursor() as cur:
                cur.execute(sql.replace("?", "%s"), params if params else ())


class Database:
    """
    Database helper class with CRUD operations.

    Attributes:
        db_path: Path to SQLite database file
        conn: Database connection (lazy-initialized)
    """

    def __init__(
        self,
        db_path: str | Path = "data/database/mlops.db",
        db_url: str | None = None,
    ) -> None:
        """
        Initialize database connection.

        Supports two backends, selected automatically:

        **PostgreSQL** (production/Docker):
            Set ``DATABASE_URL=postgresql://user:pass@host:5432/dbname`` in the
            environment, or pass ``db_url`` explicitly.  Requires psycopg2-binary
            (already listed in project dependencies).

        **SQLite** (development/tests):
            Pass ``db_path`` (or leave as default).  Falls back to SQLite when
            ``DATABASE_URL`` is absent or does not start with ``postgresql``.

        Args:
            db_path: Path to SQLite database file (default: ``data/database/mlops.db``).
                     Ignored when PostgreSQL is used.
            db_url:  PostgreSQL connection URL.  When ``None`` the environment
                     variable ``DATABASE_URL`` is checked.

        Note:
            Schema is bootstrapped automatically if the database is empty.
        """
        url: str = db_url or os.environ.get("DATABASE_URL", "")

        # Build PostgreSQL URL from individual POSTGRES_* env vars when
        # DATABASE_URL is absent.  This covers the Streamlit host process
        # which receives POSTGRES_HOST / POSTGRES_PORT / … from the Makefile
        # but NOT a full DATABASE_URL.
        if not url or not url.startswith("postgresql"):
            pg_host = os.environ.get("POSTGRES_HOST", "")
            if pg_host:
                pg_port = os.environ.get("POSTGRES_PORT", "5432")
                pg_user = os.environ.get("POSTGRES_USER", "mlops_user")
                pg_pass = os.environ.get("POSTGRES_PASSWORD", "changeme")
                pg_db = os.environ.get("POSTGRES_DB", "mlops_db")
                url = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

        # Support sqlite:///path or sqlite://path URLs — extract the path and
        # fall through to the SQLite branch below.
        if url.startswith("sqlite://"):
            # sqlite:///absolute/path or sqlite://relative/path
            sqlite_path = url[len("sqlite:///") :]
            if not sqlite_path:
                sqlite_path = url[len("sqlite://") :]
            db_path = sqlite_path
            url = ""  # clear so we fall through to the SQLite branch

        if url and url.startswith("postgresql"):
            # ---- PostgreSQL backend ----------------------------------------
            self._backend: str = "postgres"
            self.db_path: Path | None = None  # not used for PostgreSQL
            _raw_conn = psycopg2.connect(url)
            _raw_conn.autocommit = False
            self.conn: Any = _PgConnectionWrapper(_raw_conn)
            # Bootstrap schema (idempotent — uses CREATE TABLE IF NOT EXISTS).
            # Guard with a module-level set so the DDL connection is only
            # opened ONCE per URL per process, not on every Database() call.
            # This prevents exhausting PostgreSQL max_connections when many
            # Database() instances are created in a short time (e.g. Streamlit).
            if url not in _pg_bootstrapped_urls:
                import time as _time

                for _attempt in range(3):
                    try:
                        create_tables_postgres(url)
                        _pg_bootstrapped_urls.add(url)
                        break
                    except Exception as _exc:  # noqa: BLE001
                        if _attempt == 2:
                            raise
                        _time.sleep(0.5 * (_attempt + 1))
        else:
            # ---- SQLite backend (existing behaviour \u2014 unchanged) -----------
            self._backend = "sqlite"
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create tables if database doesn\u2019t exist
            if not self.db_path.exists():
                create_tables(self.db_path)

            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access

            # Run incremental migrations for existing databases (idempotent)
            _migrate_drift_signals_raw_columns_sqlite(self.conn)
            _migrate_model_approvals_challenger_test_sqlite(self.conn)
            _migrate_model_training_data_sqlite(self.conn)

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # ========================================
    # Device Management
    # ========================================

    def register_device(
        self,
        device_id: str | None = None,
        device_name: str | None = None,
        device_type: str | None = None,
        location: str | None = None,
        status: str = "active",
        deployment_mode: str = "local",
    ) -> str:
        """
        Register a new device or update existing device.

        Args:
            device_id: UUID string (auto-generated if None)
            device_name: Device name (e.g., "Device-Alpha-001")
            device_type: Device type (e.g., "Sensor-A")
            location: Location (e.g., "Building-3-Floor-2")
            status: Device status (active, inactive, maintenance)
            deployment_mode: 'local' or 'cloud' — tags the device's origin
                environment so local sandbox devices are excluded from
                production data exports.

        Returns:
            device_id (UUID string)

        Raises:
            ValueError: If status is invalid
        """
        if status not in ("active", "inactive", "maintenance"):
            raise ValueError(f"Invalid status: {status}")

        if device_id is None:
            device_id = generate_device_id()

        now = datetime.now(timezone.utc).isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO devices (device_id, device_name, device_type, location, status,
                                 first_seen_at, last_seen_at, deployment_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                device_name = COALESCE(excluded.device_name, devices.device_name),
                device_type = COALESCE(excluded.device_type, devices.device_type),
                location = COALESCE(excluded.location, devices.location),
                status = excluded.status,
                last_seen_at = excluded.last_seen_at,
                updated_at = CURRENT_TIMESTAMP
            """,
            (device_id, device_name, device_type, location, status, now, now, deployment_mode),
        )
        self.conn.commit()

        return device_id

    def get_device(self, device_id: str) -> dict[str, Any] | None:
        """
        Get device information.

        Args:
            device_id: Device UUID

        Returns:
            Device dict or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def update_device_status(self, device_id: str, status: str) -> None:
        """
        Update device status.

        Args:
            device_id: Device UUID
            status: New status (active, inactive, maintenance)

        Raises:
            ValueError: If device not found or invalid status
        """
        if status not in ("active", "inactive", "maintenance"):
            raise ValueError(f"Invalid status: {status}")

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE devices
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE device_id = ?
            """,
            (status, device_id),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Device not found: {device_id}")

        self.conn.commit()

    # ========================================
    # Prediction Storage (Atomic Transaction)
    # ========================================

    def store_prediction(
        self,
        device_id: str,
        time_values: list[float],
        amplitude_values: list[float],
        predicted_label: int,
        model_version: str,
        features: dict[str, float | None],
        prediction_confidence: float | None = None,
        shape_type: str | None = None,
        estimated_mu: float | None = None,
        estimated_sigma: float | None = None,
        mlflow_run_id: str | None = None,
        git_sha: str | None = None,
        dvc_data_hash: str | None = None,
        airflow_run_id: str | None = None,
        deployment_mode: str = "local",
    ) -> int:
        """
        Store prediction, raw signal, and features atomically.

        Args:
            device_id: Device UUID
            time_values: Time array (e.g., [0.0, 1.0, ..., 100.0])
            amplitude_values: Amplitude array (may contain NaN as None)
            predicted_label: Predicted label (0=healthy, 1=unhealthy)
            model_version: Model version (e.g., "bootstrap_v1")
            features: Feature dict from feature_extractor.extract_features()
            prediction_confidence: Model confidence [0,1]
            shape_type: Optional shape type ("gaussian", "lorentzian", or None)
            estimated_mu: Optional estimated mean
            estimated_sigma: Optional estimated std dev
            mlflow_run_id: MLflow run ID of the model's training run
            git_sha: Git commit hash of the code that trained the model
            dvc_data_hash: DVC hash of the training data
            airflow_run_id: Airflow DAG run ID (if triggered by Airflow)

        Returns:
            prediction_id

        Raises:
            ValueError: If validation fails
            sqlite3.Error: If database operation fails
        """
        # Validate inputs
        if predicted_label not in (0, 1):
            raise ValueError(f"Invalid predicted_label: {predicted_label}")

        if len(time_values) != len(amplitude_values):
            raise ValueError("time_values and amplitude_values must have same length")

        if len(time_values) < 51:
            raise ValueError(f"Signal too short: {len(time_values)} < 51")

        # Count NaN values
        n_nan = sum(1 for val in amplitude_values if val is None)
        max_nan = int(len(amplitude_values) * 0.05)
        if n_nan > max_nan:
            raise ValueError(f"Too many NaN values: {n_nan} > {max_nan} (5%)")

        # Compute metadata
        valid_amplitudes = [val for val in amplitude_values if val is not None]
        amplitude_min = min(valid_amplitudes) if valid_amplitudes else None
        amplitude_max = max(valid_amplitudes) if valid_amplitudes else None

        # Convert to JSON
        time_json = json.dumps(time_values)
        amplitude_json = json.dumps(amplitude_values)  # None represents NaN

        # Atomic transaction
        cursor = self.conn.cursor()
        try:
            # 1. Insert prediction
            timestamp = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                INSERT INTO predictions (
                    device_id, timestamp, predicted_label, prediction_confidence,
                    model_version, mlflow_run_id, git_sha, dvc_data_hash, airflow_run_id,
                    deployment_mode
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING prediction_id
                """,
                (
                    device_id,
                    timestamp,
                    predicted_label,
                    prediction_confidence,
                    model_version,
                    mlflow_run_id,
                    git_sha,
                    dvc_data_hash,
                    airflow_run_id,
                    deployment_mode,
                ),
            )
            prediction_id = cursor.fetchone()["prediction_id"]

            # 2. Insert raw signal
            cursor.execute(
                """
                INSERT INTO raw_signals (
                    prediction_id, time_values, amplitude_values,
                    n_points, n_nan_values, time_min, time_max, amplitude_min, amplitude_max,
                    shape_type, deployment_mode
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    time_json,
                    amplitude_json,
                    len(time_values),
                    n_nan,
                    min(time_values),
                    max(time_values),
                    amplitude_min,
                    amplitude_max,
                    shape_type,
                    deployment_mode,
                ),
            )

            # 3. Insert features (synchronized with feature_extractor.py)
            cursor.execute(
                """
                INSERT INTO features (
                    prediction_id, fwhm, peak_height, peak_area, noise_level,
                    snr, peak_center, estimated_mu, estimated_sigma, deployment_mode
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    features.get("fwhm"),
                    features.get("peak_height"),
                    features.get("peak_area"),
                    features.get("noise_level"),
                    features.get("snr"),
                    features.get("peak_center"),
                    estimated_mu,
                    estimated_sigma,
                    deployment_mode,
                ),
            )

            self.conn.commit()
            return prediction_id  # type: ignore[return-value]

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to store prediction: {e}") from e

    # ========================================
    # Local sandbox data wipe
    # ========================================

    def delete_local_data(self) -> dict[str, int]:
        """Delete all rows tagged with deployment_mode='local' from every table.

        Each table now carries its own ``deployment_mode`` column, so we
        delete directly per-table in FK-safe order.

        .. deprecated::
            With Phase 2 (separate database names), local and cloud data live
            in physically separate databases (``mlops_local`` vs ``mlops_prod``).
            The preferred way to reset the local sandbox is ``make reset-local-db``
            which drops and recreates the entire ``mlops_local`` database.
            This method remains functional for programmatic or test use, and for
            single-database setups where ``delete_local_data()`` is still meaningful.

        Returns:
            Dict with counts of deleted rows per table.
        """
        cursor = self.conn.cursor()
        try:
            counts: dict[str, int] = {}
            for table in (
                "sparse_labels",
                "features",
                "raw_signals",
                "predictions",
                "drift_signals",
                "drift_batches",
                "devices",
            ):
                cursor.execute(
                    f"DELETE FROM {table} WHERE deployment_mode = ?",  # noqa: S608
                    ("local",),
                )
                counts[table] = cursor.rowcount

            self.conn.commit()
            return counts

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to delete local data: {e}") from e

    # ========================================
    # Drift Provocation Persistence
    # ========================================

    def store_drift_batch(
        self,
        drift_type: str,
        n_reference: int,
        n_drifted: int,
        parameters: dict[str, Any],
        reference_rows: list[dict[str, Any]],
        drifted_rows: list[dict[str, Any]],
        ks_results: list[dict[str, Any]] | None = None,
        n_drifted_features: int | None = None,
        deployment_mode: str = "local",
    ) -> int:
        """Store a drift provocation batch with all reference and drifted signals.

        Args:
            drift_type: One of data_drift, concept_drift, feature_drift, prior_probability_drift.
            n_reference: Number of reference signals generated.
            n_drifted: Number of drifted signals generated.
            parameters: Dict of drift knobs (mu_offset, noise_multiplier, etc.).
            reference_rows: Feature dicts from ``generate_batch()`` for the reference set.
            drifted_rows: Feature dicts from ``generate_batch()`` for the drifted set.
            ks_results: Optional list of KS-test result dicts.
            n_drifted_features: Count of features that showed significant drift.
            deployment_mode: 'local' or 'cloud'.

        Returns:
            batch_id of the stored drift batch.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO drift_batches (
                    drift_type, n_reference, n_drifted, parameters,
                    n_drifted_features, ks_results, deployment_mode
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                RETURNING batch_id
                """,
                (
                    drift_type,
                    n_reference,
                    n_drifted,
                    json.dumps(parameters),
                    n_drifted_features,
                    json.dumps(ks_results) if ks_results else None,
                    deployment_mode,
                ),
            )
            batch_id: int = cursor.fetchone()["batch_id"]

            _feat_cols = ("fwhm", "peak_height", "peak_area", "noise_level", "snr", "peak_center")

            insert_sql = """
                INSERT INTO drift_signals (
                    batch_id, batch_role, fwhm, peak_height, peak_area,
                    noise_level, snr, peak_center, shape_type, label,
                    time_values, amplitude_values, predicted_label
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            for row in reference_rows:
                tv = row.get("time_values")
                av = row.get("amplitude_values")
                if tv is not None and hasattr(tv, "tolist"):
                    tv = tv.tolist()
                if av is not None and hasattr(av, "tolist"):
                    av = av.tolist()
                cursor.execute(
                    insert_sql,
                    (
                        batch_id,
                        "reference",
                        *(row.get(c) for c in _feat_cols),
                        row.get("shape_type"),
                        row.get("label"),
                        json.dumps(tv) if tv is not None else None,
                        json.dumps(av) if av is not None else None,
                        row.get("predicted_label"),
                    ),
                )

            for row in drifted_rows:
                tv = row.get("time_values")
                av = row.get("amplitude_values")
                if tv is not None and hasattr(tv, "tolist"):
                    tv = tv.tolist()
                if av is not None and hasattr(av, "tolist"):
                    av = av.tolist()
                cursor.execute(
                    insert_sql,
                    (
                        batch_id,
                        "drifted",
                        *(row.get(c) for c in _feat_cols),
                        row.get("shape_type"),
                        row.get("label"),
                        json.dumps(tv) if tv is not None else None,
                        json.dumps(av) if av is not None else None,
                        row.get("predicted_label"),
                    ),
                )

            self.conn.commit()
            return batch_id

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to store drift batch: {e}") from e

    # ========================================
    # Sparse Label Injection
    # ========================================

    def inject_sparse_label(
        self,
        prediction_id: int,
        ground_truth_label: int,
        label_source: str = "manual",
        injected_by: str | None = None,
        deployment_mode: str = "local",
    ) -> int:
        """
        Inject sparse label for a prediction.

        Automatically updates predictions.ground_truth_label via trigger.

        Args:
            prediction_id: Prediction ID
            ground_truth_label: Ground truth label (0=healthy, 1=unhealthy)
            label_source: Source of label ("manual", "automated_test", etc.)
            injected_by: User/system identifier
            deployment_mode: Deployment mode tag ("local" or "cloud")

        Returns:
            label_id

        Raises:
            ValueError: If validation fails
        """
        if ground_truth_label not in (0, 1):
            raise ValueError(f"Invalid ground_truth_label: {ground_truth_label}")

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO sparse_labels (
                prediction_id, ground_truth_label, label_source, injected_by, deployment_mode
            )
            VALUES (?, ?, ?, ?, ?)
            RETURNING label_id
            """,
            (prediction_id, ground_truth_label, label_source, injected_by, deployment_mode),
        )
        label_id: int = cursor.fetchone()["label_id"]

        # Explicitly sync predictions.ground_truth_label for both SQLite and
        # PostgreSQL (SQLite also has a trigger, but an explicit UPDATE is
        # harmless and ensures correctness in both backends).
        cursor.execute(
            """
            UPDATE predictions
            SET ground_truth_label = ?,
                label_source = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE prediction_id = ?
            """,
            (ground_truth_label, label_source, prediction_id),
        )
        self.conn.commit()
        return label_id  # type: ignore[return-value]

    def wipe_test_data(self, label_source: str = "automated_test") -> int:
        """
        Delete all rows created with a specific label_source.

        Removes matching rows from ``predictions``, ``raw_signals``,
        ``features``, and ``sparse_labels``.  Production / manually-labelled
        rows are **not** affected.

        Args:
            label_source: Label source tag to target (default: ``"automated_test"``).

        Returns:
            Number of prediction rows deleted.
        """
        cursor = self.conn.cursor()

        # Collect prediction_ids with matching label_source
        cursor.execute(
            "SELECT prediction_id FROM predictions WHERE label_source = ?",
            (label_source,),
        )
        rows = cursor.fetchall()
        ids = [r["prediction_id"] if hasattr(r, "__getitem__") else r[0] for r in rows]

        if not ids:
            return 0

        placeholders = ", ".join("?" * len(ids))

        cursor.execute(
            f"DELETE FROM sparse_labels WHERE prediction_id IN ({placeholders})",  # noqa: S608
            ids,
        )
        cursor.execute(
            f"DELETE FROM features WHERE prediction_id IN ({placeholders})",  # noqa: S608
            ids,
        )
        cursor.execute(
            f"DELETE FROM raw_signals WHERE prediction_id IN ({placeholders})",  # noqa: S608
            ids,
        )
        cursor.execute(
            f"DELETE FROM predictions WHERE prediction_id IN ({placeholders})",  # noqa: S608
            ids,
        )
        self.conn.commit()
        return len(ids)

    def wipe_all_data(self) -> dict[str, int]:
        """
        Delete ALL rows from every data table (destructive reset).

        Uses DELETE (not TRUNCATE) to avoid acquiring ACCESS EXCLUSIVE locks
        that would block concurrent readers (postgres_exporter, predictions, etc.)
        and cause indefinite lock-queue pile-ups.  Sequences are reset
        separately via ALTER SEQUENCE … RESTART so IDs restart from 1.

        Returns:
            Dict mapping table name to number of rows deleted.
        """
        cursor = self.conn.cursor()
        counts: dict[str, int] = {}
        _tables = (
            "drift_signals",
            "drift_batches",
            "sparse_labels",
            "features",
            "rescoring_runs",
            "model_approvals",
            "raw_signals",
            "predictions",
            "devices",
        )
        # Count rows before deletion
        for table in _tables:
            cursor.execute(f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608
            row = cursor.fetchone()
            counts[table] = int(row["n"] if row else 0)

        # DELETE instead of TRUNCATE: needs only ROW EXCLUSIVE (not ACCESS
        # EXCLUSIVE), so concurrent SELECTs from postgres_exporter and the
        # API model-poller don't cause an indefinite lock queue.
        for table in _tables:
            cursor.execute(f"DELETE FROM {table}")  # noqa: S608

        if self._backend == "postgres":
            # Reset sequences so next inserts start from id=1 again.
            _sequences = {
                "drift_batches": "drift_batches_batch_id_seq",
                "drift_signals": "drift_signals_signal_id_seq",
                "features": "features_feature_id_seq",
                "predictions": "predictions_prediction_id_seq",
                "raw_signals": "raw_signals_signal_id_seq",
                "sparse_labels": "sparse_labels_label_id_seq",
                "model_approvals": "model_approvals_id_seq",
                "rescoring_runs": "rescoring_runs_id_seq",
            }
            for seq in _sequences.values():
                cursor.execute(  # noqa: S608
                    f"ALTER SEQUENCE IF EXISTS {seq} RESTART WITH 1"
                )

        self.conn.commit()
        return counts

    # ========================================
    # Query Operations
    # ========================================

    def get_prediction(self, prediction_id: int) -> dict[str, Any] | None:
        """
        Get prediction with signal and features.

        Args:
            prediction_id: Prediction ID

        Returns:
            Dict with prediction, signal, features, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                p.*,
                r.time_values, r.amplitude_values, r.n_points, r.n_nan_values,
                r.shape_type,
                f.fwhm, f.peak_height, f.peak_area, f.noise_level, f.snr, f.peak_center,
                f.estimated_mu, f.estimated_sigma
            FROM predictions p
            LEFT JOIN raw_signals r ON p.prediction_id = r.prediction_id
            LEFT JOIN features f ON p.prediction_id = f.prediction_id
            WHERE p.prediction_id = ?
            """,
            (prediction_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        result = dict(row)

        # Parse JSON arrays
        if result["time_values"]:
            result["time_values"] = json.loads(result["time_values"])
        if result["amplitude_values"]:
            result["amplitude_values"] = json.loads(result["amplitude_values"])

        return result

    def get_predictions_by_device(self, device_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get recent predictions for a device.

        Args:
            device_id: Device UUID
            limit: Maximum number of predictions to return

        Returns:
            List of prediction dicts (most recent first)
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM predictions
            WHERE device_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (device_id, limit),
        )

        return [dict(row) for row in cursor.fetchall()]

    def calculate_realized_accuracy(
        self, lookback_days: int = 30, model_version: str | None = None
    ) -> dict[str, Any]:
        """
        Calculate realized accuracy from labeled predictions.

        Args:
            lookback_days: Number of days to look back
            model_version: Optional model version filter

        Returns:
            Dict with accuracy metrics
        """
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        cursor = self.conn.cursor()

        # Build query with optional model version filter
        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN predicted_label = ground_truth_label THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN ground_truth_label = 0 THEN 1 ELSE 0 END) as total_healthy,
                SUM(CASE WHEN ground_truth_label = 1 THEN 1 ELSE 0 END) as total_unhealthy,
                SUM(CASE WHEN predicted_label = 0 AND ground_truth_label = 0 THEN 1 ELSE 0 END) as true_healthy,
                SUM(CASE WHEN predicted_label = 1 AND ground_truth_label = 1 THEN 1 ELSE 0 END) as true_unhealthy
            FROM predictions
            WHERE ground_truth_label IS NOT NULL
              AND timestamp >= ?
        """
        params = [cutoff_date]

        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)

        cursor.execute(query, params)
        row = cursor.fetchone()

        total = row["total"]
        correct = row["correct"] or 0
        total_healthy = row["total_healthy"] or 0
        total_unhealthy = row["total_unhealthy"] or 0
        true_healthy = row["true_healthy"] or 0
        true_unhealthy = row["true_unhealthy"] or 0

        accuracy = correct / total if total > 0 else 0.0
        healthy_accuracy = true_healthy / total_healthy if total_healthy > 0 else 0.0
        unhealthy_accuracy = true_unhealthy / total_unhealthy if total_unhealthy > 0 else 0.0

        return {
            "lookback_days": lookback_days,
            "model_version": model_version,
            "total_labeled": total,
            "total_correct": correct,
            "accuracy": accuracy,
            "healthy_accuracy": healthy_accuracy,
            "unhealthy_accuracy": unhealthy_accuracy,
            "total_healthy": total_healthy,
            "total_unhealthy": total_unhealthy,
        }

    def get_label_coverage(self, lookback_days: int = 30) -> dict[str, Any]:
        """
        Calculate label coverage (percentage of predictions with ground truth).

        Args:
            lookback_days: Number of days to look back

        Returns:
            Dict with coverage metrics
        """
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_predictions,
                SUM(CASE WHEN ground_truth_label IS NOT NULL THEN 1 ELSE 0 END) as labeled_predictions,
                SUM(CASE WHEN predicted_label = 0 THEN 1 ELSE 0 END) as healthy_predictions,
                SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as unhealthy_predictions
            FROM predictions
            WHERE timestamp >= ?
            """,
            (cutoff_date,),
        )
        row = cursor.fetchone()

        total = row["total_predictions"]
        labeled = row["labeled_predictions"] or 0
        coverage = labeled / total if total > 0 else 0.0
        healthy_predictions = row["healthy_predictions"] or 0
        unhealthy_predictions = row["unhealthy_predictions"] or 0

        return {
            "lookback_days": lookback_days,
            "total_predictions": total,
            "labeled_predictions": labeled,
            "label_coverage": coverage,
            "healthy_predictions": healthy_predictions,
            "unhealthy_predictions": unhealthy_predictions,
        }

    # ========================================
    # Data Export Methods (DagsHub Sync)
    # ========================================

    def export_predictions_to_csv(
        self, output_path: str | Path, since: datetime | None = None
    ) -> int:
        """
        Export predictions table to CSV for DagsHub synchronization.

        Args:
            output_path: Path to output CSV file
            since: Optional datetime to export only recent predictions

        Returns:
            Number of rows exported

        Raises:
            IOError: If file cannot be written
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cursor = self.conn.cursor()

        query = """
            SELECT
                prediction_id,
                device_id,
                timestamp,
                predicted_label,
                prediction_confidence,
                model_version,
                ground_truth_label,
                label_source,
                created_at,
                updated_at
            FROM predictions
            WHERE deployment_mode = 'cloud'
        """
        params = []

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Define fieldnames (always write header even if no rows)
        fieldnames = [
            "prediction_id",
            "device_id",
            "timestamp",
            "predicted_label",
            "prediction_confidence",
            "model_version",
            "ground_truth_label",
            "label_source",
            "created_at",
            "updated_at",
        ]

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        return len(rows)

    def export_features_to_csv(self, output_path: str | Path, since: datetime | None = None) -> int:
        """
        Export features table to CSV for DagsHub synchronization.

        Args:
            output_path: Path to output CSV file
            since: Optional datetime to export only recent features

        Returns:
            Number of rows exported

        Raises:
            IOError: If file cannot be written
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cursor = self.conn.cursor()

        query = """
            SELECT
                f.feature_id,
                f.prediction_id,
                f.fwhm,
                f.peak_height,
                f.peak_area,
                f.noise_level,
                f.snr,
                f.peak_center,
                f.estimated_mu,
                f.estimated_sigma,
                f.created_at
            FROM features f
            JOIN predictions p ON f.prediction_id = p.prediction_id
            WHERE p.deployment_mode = 'cloud'
        """
        params = []

        if since:
            query += " AND p.timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY f.feature_id ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Define fieldnames (always write header even if no rows)
        fieldnames = [
            "feature_id",
            "prediction_id",
            "fwhm",
            "peak_height",
            "peak_area",
            "noise_level",
            "snr",
            "peak_center",
            "estimated_mu",
            "estimated_sigma",
            "created_at",
        ]

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        return len(rows)

    def export_devices_to_csv(self, output_path: str | Path) -> int:
        """
        Export devices table to CSV for DagsHub synchronization.

        Args:
            output_path: Path to output CSV file

        Returns:
            Number of rows exported

        Raises:
            IOError: If file cannot be written
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                device_id,
                device_name,
                device_type,
                location,
                status,
                first_seen_at,
                last_seen_at,
                created_at,
                updated_at
            FROM devices
            WHERE deployment_mode = 'cloud'
            ORDER BY first_seen_at ASC
            """
        )
        rows = cursor.fetchall()

        # Define fieldnames (always write header even if no rows)
        fieldnames = [
            "device_id",
            "device_name",
            "device_type",
            "location",
            "status",
            "first_seen_at",
            "last_seen_at",
            "created_at",
            "updated_at",
        ]

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        return len(rows)

    def export_sparse_labels_to_csv(
        self, output_path: str | Path, since: datetime | None = None
    ) -> int:
        """
        Export sparse_labels table to CSV for DagsHub synchronization.

        Args:
            output_path: Path to output CSV file
            since: Optional datetime to export only recent labels

        Returns:
            Number of rows exported

        Raises:
            IOError: If file cannot be written
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cursor = self.conn.cursor()

        query = """
            SELECT
                label_id,
                prediction_id,
                ground_truth_label,
                label_source,
                injected_at,
                injected_by
            FROM sparse_labels
            WHERE deployment_mode = 'cloud'
        """
        params = []

        if since:
            query += " AND injected_at >= ?"
            params.append(since.isoformat())

        query += " ORDER BY injected_at ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Define fieldnames (always write header even if no rows)
        fieldnames = [
            "label_id",
            "prediction_id",
            "ground_truth_label",
            "label_source",
            "injected_at",
            "injected_by",
        ]

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        return len(rows)

    @staticmethod
    def _get_shard_path(device_id: str) -> tuple[str, str]:
        """
        Extract shard prefixes from device UUID for hierarchical storage.

        Uses 2-character prefix sharding to distribute data across 256 × 256 = 65,536
        possible shards, preventing file system performance degradation.

        Args:
            device_id: UUID string (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)

        Returns:
            Tuple of (prefix1, prefix2) where:
            - prefix1: First 2 characters of UUID (hex)
            - prefix2: Characters 2-4 of UUID (hex)

        Example:
            device_id = "550e8400-e29b-41d4-a716-446655440000"
            Returns: ("55", "0e")
            Path: data/raw_signals/55/0e/550e8400-.../pred_123.json

        Rationale:
            - Uniform Distribution: UUIDs are random, ensuring balanced shards
            - Scalability: 65,536 shards support 10M+ devices efficiently
            - OS Performance: Avoids >10,000 files per directory (ext4/APFS limit)
            - DVC Efficiency: Enables parallel hash computation and caching
        """
        # Remove hyphens and take first 4 hex characters
        uuid_hex = device_id.replace("-", "")
        prefix1 = uuid_hex[:2]  # Characters 0-1
        prefix2 = uuid_hex[2:4]  # Characters 2-3
        return prefix1, prefix2

    def export_signals_to_json(self, output_dir: str | Path, signal_ids: list[int]) -> int:
        """
        Export raw signals as individual JSON files for DagsHub synchronization.

        Uses 2-level UUID prefix sharding for scalable storage:
        {output_dir}/{prefix1}/{prefix2}/{device_id}/{prediction_id}.json

        Args:
            output_dir: Directory to write JSON files
            signal_ids: List of signal_id values to export

        Returns:
            Number of signals exported

        Raises:
            IOError: If files cannot be written

        Example:
            device_id = "550e8400-e29b-41d4-a716-446655440000"
            Output: data/raw_signals/55/0e/550e8400-.../123.json
        """
        output_dir = Path(output_dir)

        cursor = self.conn.cursor()

        exported_count = 0
        for signal_id in signal_ids:
            cursor.execute(
                """
                SELECT
                    s.signal_id,
                    s.prediction_id,
                    s.time_values,
                    s.amplitude_values,
                    s.n_points,
                    s.n_nan_values,
                    s.time_min,
                    s.time_max,
                    s.amplitude_min,
                    s.amplitude_max,
                    s.shape_type,
                    s.created_at,
                    p.device_id,
                    p.timestamp AS prediction_timestamp,
                    p.predicted_label,
                    p.ground_truth_label
                FROM raw_signals s
                JOIN predictions p ON s.prediction_id = p.prediction_id
                WHERE s.signal_id = ?
                """,
                (signal_id,),
            )
            row = cursor.fetchone()

            if row:
                device_id = row["device_id"]
                prediction_id = row["prediction_id"]

                # Get shard prefixes for hierarchical storage
                prefix1, prefix2 = self._get_shard_path(device_id)

                # Create sharded directory structure: output_dir/prefix1/prefix2/device_id/
                device_dir = output_dir / prefix1 / prefix2 / device_id
                device_dir.mkdir(parents=True, exist_ok=True)

                # Create JSON file: {prediction_id}.json
                json_path = device_dir / f"{prediction_id}.json"

                signal_data = {
                    "signal_id": row["signal_id"],
                    "prediction_id": row["prediction_id"],
                    "device_id": row["device_id"],
                    "time_values": json.loads(row["time_values"]),
                    "amplitude_values": json.loads(row["amplitude_values"]),
                    "n_points": row["n_points"],
                    "n_nan_values": row["n_nan_values"],
                    "time_min": row["time_min"],
                    "time_max": row["time_max"],
                    "amplitude_min": row["amplitude_min"],
                    "amplitude_max": row["amplitude_max"],
                    "shape_type": row["shape_type"],
                    "created_at": row["created_at"].isoformat()
                    if hasattr(row["created_at"], "isoformat")
                    else row["created_at"],
                    "prediction_timestamp": row["prediction_timestamp"].isoformat()
                    if hasattr(row["prediction_timestamp"], "isoformat")
                    else row["prediction_timestamp"],
                    "predicted_label": row["predicted_label"],
                    "ground_truth_label": row["ground_truth_label"],
                }

                with open(json_path, "w") as f:
                    json.dump(signal_data, f, indent=2)

                exported_count += 1

        return exported_count

    def get_labeled_signal_ids(
        self,
        limit: int | None = None,
        window_days: int | None = None,
        signal_ids_filter: list[int] | None = None,
    ) -> list[int]:
        """
        Get signal IDs for predictions that have ground truth labels.

        Args:
            limit:             Maximum number of IDs to return.  ``None`` returns all.
            window_days:       When set, only return signals whose ``raw_signals.created_at``
                               is within the last *window_days* days.  The filter is applied
                               in SQL — no post-hoc pandas filtering required.
            signal_ids_filter: When set, restrict results to this explicit list of
                               signal_ids (used by bootstrap/champion-challenger to train
                               only on the signals that were just inserted).

        Returns:
            List of signal_id values where ground_truth_label IS NOT NULL
        """
        cursor = self.conn.cursor()
        params: list = []
        query = """
            SELECT s.signal_id
            FROM raw_signals s
            JOIN predictions p ON s.prediction_id = p.prediction_id
            WHERE p.ground_truth_label IS NOT NULL
        """
        # When caller provides an explicit signal_ids_filter it already knows
        # exactly which signals it wants — skip the deployment_mode constraint
        # so bootstrap (which may use non-cloud mode) works correctly.
        if signal_ids_filter is None:
            query += "  AND p.deployment_mode = 'cloud'\n"
        if window_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
            query += "  AND s.created_at >= ?\n"
            params.append(cutoff)
        if signal_ids_filter is not None:
            placeholders = ",".join(["?"] * len(signal_ids_filter))
            query += f"  AND s.signal_id IN ({placeholders})\n"
            params.extend(signal_ids_filter)
        query += "ORDER BY s.signal_id ASC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        cursor.execute(query, params)
        return [row["signal_id"] for row in cursor.fetchall()]

    def get_unlabeled_signal_ids(
        self,
        window_days: int | None = None,
        signal_ids_filter: list[int] | None = None,
    ) -> list[int]:
        """
        Get signal IDs for predictions without ground truth labels.

        Args:
            window_days:       When set, only return signals whose ``raw_signals.created_at``
                               is within the last *window_days* days.
            signal_ids_filter: When set, restrict results to this explicit list of
                               signal_ids.

        Returns:
            List of signal_id values where ground_truth_label IS NULL
        """
        cursor = self.conn.cursor()
        params: list = []
        query = """
            SELECT s.signal_id
            FROM raw_signals s
            JOIN predictions p ON s.prediction_id = p.prediction_id
            WHERE p.ground_truth_label IS NULL
        """
        # When caller provides an explicit signal_ids_filter it already knows
        # exactly which signals it wants — skip the deployment_mode constraint.
        if signal_ids_filter is None:
            query += "  AND p.deployment_mode = 'cloud'\n"
        if window_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
            query += "  AND s.created_at >= ?\n"
            params.append(cutoff)
        if signal_ids_filter is not None:
            placeholders = ",".join(["?"] * len(signal_ids_filter))
            query += f"  AND s.signal_id IN ({placeholders})\n"
            params.extend(signal_ids_filter)
        query += "ORDER BY s.signal_id ASC"
        cursor.execute(query, params)
        return [row["signal_id"] for row in cursor.fetchall()]

    def get_signal_id_by_prediction_id(self, prediction_id: int) -> int | None:
        """
        Return the ``signal_id`` from ``raw_signals`` for a given ``prediction_id``.

        Used by bootstrap / champion-challenger flows to map newly stored predictions
        back to signal_ids for ``record_training_split``.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT signal_id FROM raw_signals WHERE prediction_id = ?",
            (prediction_id,),
        )
        row = cursor.fetchone()
        return int(row["signal_id"]) if row else None

    def update_model_version_by_run_id(self, mlflow_run_id: str, new_model_version: str) -> int:
        """
        Update the ``model_version`` column in both ``predictions`` and
        ``model_training_data`` for all rows associated with *mlflow_run_id*.

        Called after ``_step_register()`` in the bootstrap flow so that the
        canonical version label (e.g. ``"v1"``) replaces the temporary
        run-ID-based placeholder that was used during signal storage.

        Returns:
            Total number of rows updated across both tables.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE predictions SET model_version = ? WHERE mlflow_run_id = ?",
            (new_model_version, mlflow_run_id),
        )
        n_predictions = cursor.rowcount if cursor.rowcount is not None else 0
        cursor.execute(
            "UPDATE model_training_data SET model_version = ? WHERE mlflow_run_id = ?",
            (new_model_version, mlflow_run_id),
        )
        n_training = cursor.rowcount if cursor.rowcount is not None else 0
        self.conn.commit()
        return n_predictions + n_training

    # ========================================
    # Count helpers (used by Airflow DAGs)
    # ========================================

    def count_labeled_signals(self) -> int:
        """Return total number of signals that have a confirmed ground-truth label."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) AS n
            FROM raw_signals s
            JOIN predictions p ON s.prediction_id = p.prediction_id
            WHERE p.ground_truth_label IS NOT NULL
            """
        )
        return int(cursor.fetchone()["n"])

    def count_all_signals(self) -> int:
        """Return total number of stored raw signals."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) AS n FROM raw_signals")
        return int(cursor.fetchone()["n"])

    def get_signal_data_by_id(self, signal_id: int) -> dict[str, Any] | None:
        """
        Load a single signal (time + amplitude arrays) by its signal_id.

        Args:
            signal_id: The raw_signals.signal_id to load.

        Returns:
            Dict with ``time_values`` and ``amplitude_values`` lists, or None.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT time_values, amplitude_values
            FROM raw_signals
            WHERE signal_id = ?
            """,
            (signal_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "time_values": json.loads(row["time_values"]),
            "amplitude_values": json.loads(row["amplitude_values"]),
        }

    # ========================================
    # Training Split Tracking (model_training_data)
    # ========================================

    def record_training_split(
        self,
        mlflow_run_id: str,
        train_signal_ids: list[int],
        test_signal_ids: list[int],
        model_version: str | None = None,
    ) -> None:
        """
        Record which signal IDs were used for training vs testing a given model run.

        Rows are inserted with ``INSERT OR IGNORE`` (SQLite) / ``ON CONFLICT DO NOTHING``
        (PostgreSQL) so calling this method twice for the same run is safe.

        Args:
            mlflow_run_id:    MLflow run ID of the training job.
            train_signal_ids: Signal IDs that were in the training split.
            test_signal_ids:  Signal IDs that were in the test (gold-standard) split.
            model_version:    Optional human-readable version tag (e.g. ``"v7"``).
        """
        cursor = self.conn.cursor()
        rows = [(mlflow_run_id, sid, "train", model_version) for sid in train_signal_ids] + [
            (mlflow_run_id, sid, "test", model_version) for sid in test_signal_ids
        ]
        if not rows:
            return
        if self._backend == "postgres":
            cursor.executemany(
                """
                INSERT INTO model_training_data (mlflow_run_id, signal_id, split, model_version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (mlflow_run_id, signal_id) DO NOTHING
                """,
                rows,
            )
        else:
            cursor.executemany(
                """
                INSERT OR IGNORE INTO model_training_data
                    (mlflow_run_id, signal_id, split, model_version)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
        self.conn.commit()

    def get_training_signal_ids(
        self,
        mlflow_run_id: str,
        split: str | None = None,
    ) -> list[int]:
        """
        Return signal IDs recorded for a given MLflow run, optionally filtered by split.

        Args:
            mlflow_run_id: MLflow run ID to query.
            split:         ``"train"``, ``"test"``, or ``None`` (returns all).

        Returns:
            List of signal_id integers.
        """
        cursor = self.conn.cursor()
        if split is not None:
            cursor.execute(
                "SELECT signal_id FROM model_training_data "
                "WHERE mlflow_run_id = ? AND split = ? ORDER BY signal_id",
                (mlflow_run_id, split),
            )
        else:
            cursor.execute(
                "SELECT signal_id FROM model_training_data "
                "WHERE mlflow_run_id = ? ORDER BY signal_id",
                (mlflow_run_id,),
            )
        return [row["signal_id"] for row in cursor.fetchall()]

    def store_features(self, signal_id: int, features: dict[str, float | None]) -> None:
        """
        Persist extracted features for an existing signal (upsert by prediction_id).

        Used by the Airflow retraining DAG after batch feature extraction.

        Args:
            signal_id: The raw_signals.signal_id whose prediction receives the features.
            features:  Dict with keys matching the ``features`` table columns
                       (fwhm, peak_height, peak_area, noise_level, snr, peak_center).
        """
        # Resolve prediction_id from signal_id
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT prediction_id FROM raw_signals WHERE signal_id = ?",
            (signal_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"signal_id {signal_id} not found")
        prediction_id: int = row["prediction_id"]

        # Upsert: delete existing features row then re-insert
        cursor.execute(
            "DELETE FROM features WHERE prediction_id = ?",
            (prediction_id,),
        )
        # Inherit deployment_mode from parent prediction
        cursor.execute(
            "SELECT deployment_mode FROM predictions WHERE prediction_id = ?",
            (prediction_id,),
        )
        dm_row = cursor.fetchone()
        dep_mode = dm_row["deployment_mode"] if dm_row else "local"
        cursor.execute(
            """
            INSERT INTO features
                (prediction_id, fwhm, peak_height, peak_area, noise_level, snr,
                 peak_center, deployment_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                features.get("fwhm"),
                features.get("peak_height"),
                features.get("peak_area"),
                features.get("noise_level"),
                features.get("snr"),
                features.get("peak_center"),
                dep_mode,
            ),
        )
        self.conn.commit()

    def get_features_by_signal_id(self, signal_id: int) -> dict[str, float | None] | None:
        """
        Return the extracted features for a signal, or ``None`` if not found.

        Args:
            signal_id: The ``raw_signals.signal_id`` to look up.

        Returns:
            Dict of feature name → value (fwhm, peak_height, peak_area,
            noise_level, snr, peak_center), or ``None`` when no features exist.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT f.fwhm, f.peak_height, f.peak_area,
                   f.noise_level, f.snr, f.peak_center
            FROM features f
            JOIN raw_signals rs ON rs.prediction_id = f.prediction_id
            WHERE rs.signal_id = ?
            """,
            (signal_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        keys = ("fwhm", "peak_height", "peak_area", "noise_level", "snr", "peak_center")
        if hasattr(row, "keys"):
            return {k: row[k] for k in keys}
        return dict(zip(keys, row))  # noqa: B905

    def get_label_by_signal_id(self, signal_id: int) -> int | None:
        """
        Return the ground-truth label for a signal, or ``None`` if not labelled.

        Args:
            signal_id: The ``raw_signals.signal_id`` to look up.

        Returns:
            ``ground_truth_label`` integer (0 or 1), or ``None`` if not labelled.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT p.ground_truth_label
            FROM predictions p
            JOIN raw_signals rs ON rs.prediction_id = p.prediction_id
            WHERE rs.signal_id = ?
            """,
            (signal_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        val = row["ground_truth_label"] if hasattr(row, "keys") else row[0]
        return int(val) if val is not None else None
