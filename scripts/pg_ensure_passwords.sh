#!/usr/bin/env bash
# =============================================================================
# pg_ensure_passwords.sh
# =============================================================================
# Idempotent password synchronisation for PostgreSQL.
#
# Reads POSTGRES_PASSWORD from the environment (same var used by the postgres
# Docker image to initialise the superuser password) and sets the password for
# POSTGRES_USER to that value.
#
# Run modes:
#   1. docker-entrypoint-initdb.d  — executed automatically on fresh volumes
#      The script is mounted at /docker-entrypoint-initdb.d/10_ensure_passwords.sh
#
#   2. On-demand (fix drift on existing volumes):
#      make fix-db-password
#      which runs:  docker exec mlops_postgres /scripts/pg_ensure_passwords.sh
#
# Why this matters:
#   The POSTGRES_PASSWORD env var only sets the password once — when the data
#   volume is first initialised.  If the password ever drifts (e.g. an ALTER
#   USER was run manually) this script brings it back in sync with .env.secrets.
# =============================================================================
set -euo pipefail

TARGET_USER="${POSTGRES_USER:-mlops_user}"
TARGET_PASSWORD="${POSTGRES_PASSWORD:-changeme}"

echo "[pg_ensure_passwords] Syncing password for user '${TARGET_USER}' ..."

psql -v ON_ERROR_STOP=1 --username "${TARGET_USER}" --no-password <<-EOSQL
    ALTER USER ${TARGET_USER} PASSWORD '${TARGET_PASSWORD}';
EOSQL

echo "[pg_ensure_passwords] Done — password is now synchronised with POSTGRES_PASSWORD."
