"""
Database initialization module.

Creates SQLite database with schema for MLOps device health monitoring:
- devices: Device registry with UUID identifiers
- predictions: Model predictions with sparse labels
- raw_signals: Variable-length time series (JSONB storage)
- features: Extracted signal features (aligned with feature_extractor.py)
- sparse_labels: Audit trail for label ingestion

Schema Design Decisions:
1. device_id uses UUID (best practice for distributed systems)
2. shape_type in raw_signals is NULLABLE (synthetic=populated, real-world=NULL)
3. estimated_mu/noise moved to features table (logical grouping)
4. Features aligned with actual feature_extractor.py output
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any


def _migrate_predictions_traceability(cursor: sqlite3.Cursor) -> None:
    """Add traceability columns to existing predictions tables (idempotent)."""
    # Query existing columns
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    new_columns = {
        "mlflow_run_id": "TEXT",
        "git_sha": "TEXT",
        "dvc_data_hash": "TEXT",
        "airflow_run_id": "TEXT",
        "deployment_mode": "TEXT DEFAULT 'local'",
    }

    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")


def _migrate_deployment_mode_all_tables(cursor: sqlite3.Cursor) -> None:
    """Add deployment_mode to raw_signals, features, sparse_labels, devices (idempotent)."""
    for table in ("raw_signals", "features", "sparse_labels", "devices"):
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        if "deployment_mode" not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN deployment_mode TEXT DEFAULT 'local'")


def _migrate_predictions_traceability_pg(cur: Any) -> None:
    """Add traceability columns to existing PostgreSQL predictions tables (idempotent)."""
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'predictions'"
    )
    existing_columns = {row[0] for row in cur.fetchall()}

    new_columns = {
        "mlflow_run_id": "TEXT",
        "git_sha": "TEXT",
        "dvc_data_hash": "TEXT",
        "airflow_run_id": "TEXT",
        "deployment_mode": "TEXT DEFAULT 'local'",
    }

    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            cur.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")


def _migrate_deployment_mode_all_tables_pg(cur: Any) -> None:
    """Add deployment_mode to raw_signals, features, sparse_labels, devices in PG (idempotent)."""
    for table in ("raw_signals", "features", "sparse_labels", "devices"):
        # Skip tables that do not exist yet (fresh database — CREATE TABLE will include the column)
        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s",
            (table,),
        )
        if not cur.fetchone():
            continue
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (table,),
        )
        existing = {row[0] for row in cur.fetchall()}
        if "deployment_mode" not in existing:
            cur.execute(
                f"ALTER TABLE {table} ADD COLUMN deployment_mode TEXT NOT NULL DEFAULT 'local'"
            )


def _migrate_drift_signals_raw_columns_pg(cur: Any) -> None:
    """Add time_values, amplitude_values, predicted_label to drift_signals (idempotent).

    These columns were added after the initial schema; existing tables created
    with ``CREATE TABLE IF NOT EXISTS`` keep the old structure.
    """
    # Skip when drift_signals does not exist yet (fresh database — CREATE TABLE includes columns)
    cur.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'drift_signals'",
    )
    if not cur.fetchone():
        return
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'drift_signals'",
    )
    existing = {row[0] for row in cur.fetchall()}
    for col, typedef in (
        ("time_values", "TEXT"),
        ("amplitude_values", "TEXT"),
        ("predicted_label", "INTEGER"),
    ):
        if col not in existing:
            cur.execute(f"ALTER TABLE drift_signals ADD COLUMN {col} {typedef}")


def _migrate_model_approvals_challenger_test_pg(cur: Any) -> None:
    """Add champion_f1_on_challenger_test column to model_approvals (idempotent, PG)."""
    cur.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'model_approvals'",
    )
    if not cur.fetchone():
        return  # Table doesn't exist yet — CREATE TABLE will include the column
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'model_approvals'",
    )
    existing = {row[0] for row in cur.fetchall()}
    if "champion_f1_on_challenger_test" not in existing:
        cur.execute("ALTER TABLE model_approvals ADD COLUMN champion_f1_on_challenger_test FLOAT")


def _migrate_model_training_data_pg(cur: Any) -> None:
    """Create model_training_data table if it doesn't exist (idempotent, PG)."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_training_data (
            id              SERIAL PRIMARY KEY,
            mlflow_run_id   TEXT NOT NULL,
            signal_id       INTEGER NOT NULL,
            split           TEXT NOT NULL,
            model_version   TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (mlflow_run_id, signal_id),
            CONSTRAINT model_training_data_split_check
                CHECK (split IN ('train', 'test'))
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_pg_mtd_run ON model_training_data(mlflow_run_id, split)"
    )


def _migrate_model_approvals_challenger_test_sqlite(conn: Any) -> None:
    """Add champion_f1_on_challenger_test to SQLite model_approvals (idempotent)."""
    import sqlite3

    try:
        cur = conn.execute("PRAGMA table_info(model_approvals)")
        existing = {row[1] for row in cur.fetchall()}
        if "champion_f1_on_challenger_test" not in existing:
            conn.execute(
                "ALTER TABLE model_approvals ADD COLUMN champion_f1_on_challenger_test REAL"
            )
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Table does not exist yet


def _migrate_model_training_data_sqlite(conn: Any) -> None:
    """Create model_training_data table in SQLite if missing (idempotent)."""
    import sqlite3

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mlflow_run_id TEXT NOT NULL,
                signal_id INTEGER NOT NULL,
                split TEXT NOT NULL,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(mlflow_run_id, signal_id),
                CHECK (split IN ('train', 'test'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mtd_run ON model_training_data(mlflow_run_id, split)"
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass


def _migrate_drift_signals_raw_columns_sqlite(conn: Any) -> None:
    """Add time_values, amplitude_values, predicted_label to SQLite drift_signals (idempotent).

    SQLite equivalent of _migrate_drift_signals_raw_columns_pg.  Used when the
    database file pre-dates the raw-signal storage columns.
    """
    import sqlite3

    try:
        cur = conn.execute("PRAGMA table_info(drift_signals)")
        existing = {row[1] for row in cur.fetchall()}  # row[1] is column name
        for col, typedef in (
            ("time_values", "TEXT"),
            ("amplitude_values", "TEXT"),
            ("predicted_label", "INTEGER"),
        ):
            if col not in existing:
                conn.execute(f"ALTER TABLE drift_signals ADD COLUMN {col} {typedef}")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Table does not exist yet — schema creation will add the columns


def create_tables(db_path: Path | str = "data/database/mlops.db") -> None:
    """
    Create all database tables with indexes and constraints.

    Args:
        db_path: Path to SQLite database file

    Raises:
        sqlite3.Error: If table creation fails
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON;")

        # ========================================
        # Table 1: devices
        # ========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,  -- UUID format (36 chars with hyphens)
                device_name TEXT,             -- e.g., "Device-Alpha-001" (faker)
                device_type TEXT,             -- e.g., "Sensor-A", "Controller-B" (faker)
                location TEXT,                -- e.g., "Building-3-Floor-2" (faker)
                status TEXT DEFAULT 'active', -- active, inactive, maintenance
                first_seen_at TEXT NOT NULL,  -- ISO 8601 timestamp
                last_seen_at TEXT,            -- ISO 8601 timestamp
                deployment_mode TEXT DEFAULT 'local',  -- 'local' or 'cloud'
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

                CHECK (device_id LIKE '________-____-____-____-____________'),  -- UUID format
                CHECK (status IN ('active', 'inactive', 'maintenance'))
            );
        """)

        # ========================================
        # Table 2: predictions
        # ========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,           -- ISO 8601 timestamp
                predicted_label INTEGER NOT NULL,  -- 0=healthy, 1=unhealthy
                prediction_confidence REAL,        -- Model confidence [0,1]
                model_version TEXT NOT NULL,       -- e.g., "bootstrap_v1", "retrained_v2"
                ground_truth_label INTEGER,        -- NULL initially, populated via sparse labeling
                label_source TEXT,                 -- NULL or "manual", "automated_test", etc.
                mlflow_run_id TEXT,                -- MLflow run ID that produced the model
                git_sha TEXT,                      -- Git commit hash of code that trained the model
                dvc_data_hash TEXT,                -- DVC hash of training data
                airflow_run_id TEXT,               -- Airflow DAG run ID (if triggered by Airflow)
                deployment_mode TEXT DEFAULT 'local',  -- 'local' or 'cloud'
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (device_id) REFERENCES devices(device_id),
                CHECK (predicted_label IN (0, 1)),
                CHECK (ground_truth_label IS NULL OR ground_truth_label IN (0, 1)),
                CHECK (prediction_confidence IS NULL OR (prediction_confidence >= 0 AND prediction_confidence <= 1))
            );
        """)

        # ========================================
        # Table 3: raw_signals
        # ========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                time_values TEXT NOT NULL,         -- JSON array: [0.0, 1.0, ..., 100.0]
                amplitude_values TEXT NOT NULL,    -- JSON array: [2.5, null, 2.7, ...] (null=NaN)
                n_points INTEGER NOT NULL,
                n_nan_values INTEGER DEFAULT 0,
                time_min REAL,
                time_max REAL,
                amplitude_min REAL,
                amplitude_max REAL,
                shape_type TEXT,                   -- NULLABLE: "gaussian", "lorentzian", or NULL (real-world)
                deployment_mode TEXT DEFAULT 'local',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
                CHECK (n_points >= 51),                           -- Minimum signal length
                CHECK (n_nan_values <= CAST(n_points * 0.05 AS INTEGER)),  -- Max 5% NaN
                CHECK (shape_type IS NULL OR shape_type IN ('gaussian', 'lorentzian'))
            );
        """)

        # ========================================
        # Table 4: features
        # ========================================
        # Synchronized with feature_extractor.py output:
        # - fwhm, peak_height, peak_area, noise_level, snr, peak_center
        # - estimated_mu, estimated_sigma (moved from raw_signals)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                fwhm REAL,                    -- Full Width Half Maximum
                peak_height REAL,             -- Maximum amplitude
                peak_area REAL,               -- Integrated area under curve
                noise_level REAL,             -- Estimated noise std dev
                snr REAL,                     -- Signal-to-Noise Ratio
                peak_center REAL,             -- Time coordinate of peak
                estimated_mu REAL,            -- Estimated mean (for Gaussian fits)
                estimated_sigma REAL,         -- Estimated std dev (for Gaussian fits)
                deployment_mode TEXT DEFAULT 'local',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
            );
        """)

        # ========================================
        # Table 5: sparse_labels
        # ========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sparse_labels (
                label_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                ground_truth_label INTEGER NOT NULL,
                label_source TEXT NOT NULL,              -- "manual", "automated_test", etc.
                injected_at TEXT DEFAULT CURRENT_TIMESTAMP,
                injected_by TEXT,                        -- User/system identifier
                deployment_mode TEXT DEFAULT 'local',

                FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
                CHECK (ground_truth_label IN (0, 1))
            );
        """)

        # ========================================
        # Table 6: drift_batches
        # ========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_batches (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                drift_type TEXT NOT NULL,
                n_reference INTEGER NOT NULL,
                n_drifted INTEGER NOT NULL,
                parameters TEXT NOT NULL,           -- JSON of drift knobs
                n_drifted_features INTEGER,         -- count of KS-significant features
                ks_results TEXT,                    -- JSON of KS-test results
                deployment_mode TEXT DEFAULT 'local',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                CHECK (drift_type IN ('data_drift','concept_drift','feature_drift','prior_probability_drift'))
            );
        """)

        # ========================================
        # Table 7: drift_signals
        # ========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                batch_role TEXT NOT NULL,          -- 'reference' or 'drifted'
                fwhm REAL,
                peak_height REAL,
                peak_area REAL,
                noise_level REAL,
                snr REAL,
                peak_center REAL,
                shape_type TEXT,
                label INTEGER,
                time_values TEXT,                  -- JSON array of raw time points
                amplitude_values TEXT,             -- JSON array of raw amplitudes
                predicted_label INTEGER,           -- model prediction (0/1)
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (batch_id) REFERENCES drift_batches(batch_id),
                CHECK (batch_role IN ('reference', 'drifted')),
                CHECK (label IN (0, 1))
            );
        """)

        # ========================================
        # Table 8: model_approvals  (Task 5 — Human Review Gate)
        # ========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                mlflow_run_id TEXT,
                challenger_f1 REAL,
                champion_f1 REAL,
                champion_f1_on_challenger_test REAL,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                decided_at TEXT,
                decided_by TEXT,

                CHECK (status IN ('pending', 'approved', 'rejected'))
            );
        """)

        # ========================================
        # Table 9: rescoring_runs  (Task 3 — Batch Re-Scoring)
        # ========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rescoring_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                rescored_at TEXT DEFAULT CURRENT_TIMESTAMP,
                n_predictions INTEGER NOT NULL,
                n_changed INTEGER DEFAULT 0,
                change_rate REAL,
                triggered_by TEXT DEFAULT 'manual',
                status TEXT DEFAULT 'completed',

                CHECK (status IN ('pending', 'running', 'completed', 'failed'))
            );
        """)

        # ========================================
        # Table 10: model_training_data  (Phase 2 — Training Split Tracking)
        # ========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mlflow_run_id TEXT NOT NULL,
                signal_id INTEGER NOT NULL,
                split TEXT NOT NULL,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(mlflow_run_id, signal_id),
                CHECK (split IN ('train', 'test'))
            );
        """)

        # ========================================
        # Indexes for Performance
        # ========================================

        # Migration: Add traceability columns to existing predictions tables
        # Must run BEFORE index creation on new columns
        _migrate_predictions_traceability(cursor)
        _migrate_deployment_mode_all_tables(cursor)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_device_timestamp "
            "ON predictions(device_id, timestamp);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(ground_truth_label);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_model_version "
            "ON predictions(model_version);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_signals_prediction ON raw_signals(prediction_id);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_prediction ON features(prediction_id);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sparse_labels_prediction "
            "ON sparse_labels(prediction_id);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_mlflow_run_id "
            "ON predictions(mlflow_run_id);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_deployment_mode "
            "ON predictions(deployment_mode);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_signals_deployment_mode "
            "ON raw_signals(deployment_mode);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_deployment_mode ON features(deployment_mode);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sparse_labels_deployment_mode "
            "ON sparse_labels(deployment_mode);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_devices_deployment_mode ON devices(deployment_mode);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_drift_batches_type ON drift_batches(drift_type);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_drift_signals_batch ON drift_signals(batch_id);"
        )

        # ========================================
        # Trigger: Update predictions.ground_truth_label on sparse label insert
        # ========================================
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_ground_truth_on_sparse_label
            AFTER INSERT ON sparse_labels
            FOR EACH ROW
            BEGIN
                UPDATE predictions
                SET ground_truth_label = NEW.ground_truth_label,
                    label_source = NEW.label_source,
                    updated_at = CURRENT_TIMESTAMP
                WHERE prediction_id = NEW.prediction_id;
            END;
        """)

        conn.commit()
        print(f"✓ Database initialized successfully: {db_path}")
        print("  Tables created: devices, predictions, raw_signals, features, sparse_labels")
        print("  Indexes created: 6 indexes for query performance")
        print("  Triggers created: 1 trigger for sparse label propagation")

    except sqlite3.Error as e:
        conn.rollback()
        raise RuntimeError(f"Failed to create database tables: {e}") from e
    finally:
        conn.close()


def verify_schema(db_path: Path | str = "data/database/mlops.db") -> dict[str, Any]:
    """
    Verify database schema is correctly created.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Dictionary with verification results
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    results = {
        "tables": [],
        "indexes": [],
        "triggers": [],
        "foreign_keys_enabled": False,
    }

    try:
        # Enable and check foreign keys
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("PRAGMA foreign_keys;")
        results["foreign_keys_enabled"] = cursor.fetchone()[0] == 1

        # Check tables (exclude sqlite_sequence which is auto-created for AUTOINCREMENT)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        results["tables"] = [row[0] for row in cursor.fetchall()]

        # Check indexes
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        results["indexes"] = [row[0] for row in cursor.fetchall()]

        # Check triggers
        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name;")
        results["triggers"] = [row[0] for row in cursor.fetchall()]

    finally:
        conn.close()

    return results


def generate_device_id() -> str:
    """
    Generate a UUID v4 for device identification.

    Returns:
        UUID string in format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# PostgreSQL schema bootstrap
# ---------------------------------------------------------------------------


def create_tables_postgres(db_url: str) -> None:
    """
    Bootstrap the PostgreSQL application schema (idempotent).

    Creates tables ``devices``, ``predictions``, ``raw_signals``,
    ``features``, and ``sparse_labels`` if they do not already exist,
    together with indexes and a trigger that propagates sparse labels to
    the ``predictions`` table.

    Args:
        db_url: psycopg2-style connection URL
                ``postgresql://user:pass@host:5432/dbname``.

    Raises:
        ImportError: If psycopg2 is not installed.
        psycopg2.Error: If the DDL statements fail.
    """
    import psycopg2

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # ---- devices -------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                device_id   VARCHAR(36) PRIMARY KEY,
                device_name TEXT,
                device_type TEXT,
                location    TEXT,
                status      TEXT NOT NULL DEFAULT 'active',
                first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_seen_at  TIMESTAMPTZ,
                deployment_mode TEXT NOT NULL DEFAULT 'local',
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT devices_status_check
                    CHECK (status IN ('active', 'inactive', 'maintenance'))
            )
        """)

        # ---- predictions ---------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id       SERIAL PRIMARY KEY,
                device_id           VARCHAR(36) NOT NULL REFERENCES devices(device_id),
                timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                predicted_label     INTEGER NOT NULL,
                prediction_confidence FLOAT,
                model_version       TEXT NOT NULL,
                ground_truth_label  INTEGER,
                label_source        TEXT,
                mlflow_run_id       TEXT,
                git_sha             TEXT,
                dvc_data_hash       TEXT,
                airflow_run_id      TEXT,
                deployment_mode     TEXT NOT NULL DEFAULT 'local',
                created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT predictions_label_check
                    CHECK (predicted_label IN (0, 1)),
                CONSTRAINT predictions_gt_check
                    CHECK (ground_truth_label IS NULL OR ground_truth_label IN (0, 1)),
                CONSTRAINT predictions_confidence_check
                    CHECK (prediction_confidence IS NULL
                           OR (prediction_confidence >= 0 AND prediction_confidence <= 1))
            )
        """)

        # Migrate existing PostgreSQL predictions tables (add traceability columns)
        _migrate_predictions_traceability_pg(cur)
        _migrate_deployment_mode_all_tables_pg(cur)
        _migrate_drift_signals_raw_columns_pg(cur)
        _migrate_model_approvals_challenger_test_pg(cur)
        _migrate_model_training_data_pg(cur)

        # ---- raw_signals ---------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_signals (
                signal_id         SERIAL PRIMARY KEY,
                prediction_id     INTEGER NOT NULL REFERENCES predictions(prediction_id),
                time_values       TEXT NOT NULL,
                amplitude_values  TEXT NOT NULL,
                n_points          INTEGER NOT NULL,
                n_nan_values      INTEGER NOT NULL DEFAULT 0,
                time_min          FLOAT,
                time_max          FLOAT,
                amplitude_min     FLOAT,
                amplitude_max     FLOAT,
                shape_type        TEXT,
                deployment_mode   TEXT NOT NULL DEFAULT 'local',
                created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT raw_signals_n_points_check  CHECK (n_points >= 51),
                CONSTRAINT raw_signals_shape_type_check
                    CHECK (shape_type IS NULL
                           OR shape_type IN ('gaussian', 'lorentzian'))
            )
        """)

        # ---- features ------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id      SERIAL PRIMARY KEY,
                prediction_id   INTEGER NOT NULL REFERENCES predictions(prediction_id),
                fwhm            FLOAT,
                peak_height     FLOAT,
                peak_area       FLOAT,
                noise_level     FLOAT,
                snr             FLOAT,
                peak_center     FLOAT,
                estimated_mu    FLOAT,
                estimated_sigma FLOAT,
                deployment_mode   TEXT NOT NULL DEFAULT 'local',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        # ---- sparse_labels -------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sparse_labels (
                label_id            SERIAL PRIMARY KEY,
                prediction_id       INTEGER NOT NULL REFERENCES predictions(prediction_id),
                ground_truth_label  INTEGER NOT NULL,
                label_source        TEXT NOT NULL,
                injected_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                injected_by         TEXT,
                deployment_mode     TEXT NOT NULL DEFAULT 'local',
                CONSTRAINT sparse_labels_gt_check
                    CHECK (ground_truth_label IN (0, 1))
            )
        """)

        # ---- drift_batches --------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS drift_batches (
                batch_id        SERIAL PRIMARY KEY,
                drift_type      TEXT NOT NULL,
                n_reference     INTEGER NOT NULL,
                n_drifted       INTEGER NOT NULL,
                parameters      TEXT NOT NULL,
                n_drifted_features INTEGER,
                ks_results      TEXT,
                deployment_mode TEXT NOT NULL DEFAULT 'local',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT drift_batches_type_check
                    CHECK (drift_type IN ('data_drift','concept_drift','feature_drift','prior_probability_drift'))
            )
        """)

        # ---- drift_signals -------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS drift_signals (
                signal_id       SERIAL PRIMARY KEY,
                batch_id        INTEGER NOT NULL REFERENCES drift_batches(batch_id),
                batch_role      TEXT NOT NULL,
                fwhm            FLOAT,
                peak_height     FLOAT,
                peak_area       FLOAT,
                noise_level     FLOAT,
                snr             FLOAT,
                peak_center     FLOAT,
                shape_type      TEXT,
                label           INTEGER,
                time_values     TEXT,
                amplitude_values TEXT,
                predicted_label INTEGER,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT drift_signals_role_check
                    CHECK (batch_role IN ('reference', 'drifted')),
                CONSTRAINT drift_signals_label_check
                    CHECK (label IN (0, 1))
            )
        """)

        # ---- model_approvals  (Task 5 — Human Review Gate) -----------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_approvals (
                id              SERIAL PRIMARY KEY,
                model_version   TEXT NOT NULL,
                mlflow_run_id   TEXT,
                challenger_f1   FLOAT,
                champion_f1     FLOAT,
                champion_f1_on_challenger_test FLOAT,
                status          TEXT NOT NULL DEFAULT 'pending',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                decided_at      TIMESTAMPTZ,
                decided_by      TEXT,
                CONSTRAINT model_approvals_status_check
                    CHECK (status IN ('pending', 'approved', 'rejected'))
            )
        """)

        # ---- rescoring_runs  (Task 3 — Batch Re-Scoring) -------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rescoring_runs (
                id              SERIAL PRIMARY KEY,
                model_version   TEXT NOT NULL,
                rescored_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                n_predictions   INTEGER NOT NULL,
                n_changed       INTEGER DEFAULT 0,
                change_rate     FLOAT,
                triggered_by    TEXT DEFAULT 'manual',
                status          TEXT NOT NULL DEFAULT 'completed',
                CONSTRAINT rescoring_runs_status_check
                    CHECK (status IN ('pending', 'running', 'completed', 'failed'))
            )
        """)

        # ---- model_training_data  (Phase 2 — Training Split Tracking) ------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_training_data (
                id              SERIAL PRIMARY KEY,
                mlflow_run_id   TEXT NOT NULL,
                signal_id       INTEGER NOT NULL,
                split           TEXT NOT NULL,
                model_version   TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (mlflow_run_id, signal_id),
                CONSTRAINT model_training_data_split_check
                    CHECK (split IN ('train', 'test'))
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pg_mtd_run ON model_training_data(mlflow_run_id, split)"
        )

        # ---- indexes -------------------------------------------------------
        for ddl in [
            "CREATE INDEX IF NOT EXISTS idx_pg_pred_device_ts "
            "ON predictions(device_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_pg_pred_label ON predictions(ground_truth_label)",
            "CREATE INDEX IF NOT EXISTS idx_pg_pred_model ON predictions(model_version)",
            "CREATE INDEX IF NOT EXISTS idx_pg_pred_mlflow_run ON predictions(mlflow_run_id)",
            "CREATE INDEX IF NOT EXISTS idx_pg_pred_deploy_mode ON predictions(deployment_mode)",
            "CREATE INDEX IF NOT EXISTS idx_pg_rawsig_pred ON raw_signals(prediction_id)",
            "CREATE INDEX IF NOT EXISTS idx_pg_rawsig_deploy_mode ON raw_signals(deployment_mode)",
            "CREATE INDEX IF NOT EXISTS idx_pg_feat_pred ON features(prediction_id)",
            "CREATE INDEX IF NOT EXISTS idx_pg_feat_deploy_mode ON features(deployment_mode)",
            "CREATE INDEX IF NOT EXISTS idx_pg_sparse_pred ON sparse_labels(prediction_id)",
            "CREATE INDEX IF NOT EXISTS idx_pg_sparse_deploy_mode ON sparse_labels(deployment_mode)",
            "CREATE INDEX IF NOT EXISTS idx_pg_devices_deploy_mode ON devices(deployment_mode)",
            "CREATE INDEX IF NOT EXISTS idx_pg_drift_batches_type ON drift_batches(drift_type)",
            "CREATE INDEX IF NOT EXISTS idx_pg_drift_signals_batch ON drift_signals(batch_id)",
        ]:
            cur.execute(ddl)

        # ---- trigger: propagate sparse label -> predictions ----------------
        cur.execute("""
            CREATE OR REPLACE FUNCTION _sync_ground_truth()
            RETURNS TRIGGER LANGUAGE plpgsql AS $$
            BEGIN
                UPDATE predictions
                SET ground_truth_label = NEW.ground_truth_label,
                    label_source       = NEW.label_source,
                    updated_at         = NOW()
                WHERE prediction_id = NEW.prediction_id;
                RETURN NEW;
            END;
            $$
        """)
        # Create trigger only if it doesn't already exist
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger
                    WHERE tgname = 'trg_sync_ground_truth'
                ) THEN
                    CREATE TRIGGER trg_sync_ground_truth
                    AFTER INSERT ON sparse_labels
                    FOR EACH ROW
                    EXECUTE FUNCTION _sync_ground_truth();
                END IF;
            END;
            $$
        """)

        conn.commit()

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    # Initialize database
    db_path = Path("data/database/mlops.db")
    create_tables(db_path)

    # Verify schema
    schema = verify_schema(db_path)
    print("\n📊 Schema Verification:")
    print(f"  Tables: {', '.join(schema['tables'])}")
    print(f"  Indexes: {len(schema['indexes'])} created")
    print(f"  Triggers: {len(schema['triggers'])} created")
    print(f"  Foreign keys enabled: {schema['foreign_keys_enabled']}")
