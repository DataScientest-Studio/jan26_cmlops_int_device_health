#!/usr/bin/env bash
# =============================================================================
# init_app_dbs.sh
# =============================================================================
# Ensures both mlops_local and mlops_prod application databases exist.
#
# The POSTGRES_DB env var creates exactly ONE database on first volume init
# (either mlops_local in local mode, or mlops_prod in cloud mode).
# This script creates the OTHER database so the volume is always ready for
# either deployment mode without needing a wipe and re-init.
#
# Mounted at: /docker-entrypoint-initdb.d/02_init_app_dbs.sh
# Runs once per fresh volume, immediately after 01_init_db.sql.
#
# On-demand use (re-run against existing volume, e.g. after a manual DB drop):
#   docker exec mlops_postgres bash /scripts/init_app_dbs.sh
# =============================================================================
set -euo pipefail

DB_USER="${POSTGRES_USER:-mlops_user}"
# Always connect to the maintenance database; mlops_user-named DB does not exist
PSQL="psql -U $DB_USER -d postgres"

echo "[init_app_dbs] Ensuring both app databases exist ..."

for DBNAME in mlops_local mlops_prod; do
    EXISTS=$($PSQL -tAc "SELECT 1 FROM pg_database WHERE datname = '$DBNAME'")
    if [ "$EXISTS" = "1" ]; then
        echo "[init_app_dbs]   '$DBNAME' already exists — skipping."
    else
        echo "[init_app_dbs]   Creating '$DBNAME' ..."
        $PSQL -c "CREATE DATABASE \"$DBNAME\" OWNER \"$DB_USER\""
        $PSQL -c "GRANT ALL PRIVILEGES ON DATABASE \"$DBNAME\" TO \"$DB_USER\""
        echo "[init_app_dbs]   '$DBNAME' created and privileges granted."
    fi
done

echo "[init_app_dbs] Done — both mlops_local and mlops_prod are ready."
