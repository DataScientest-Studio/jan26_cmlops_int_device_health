-- Initialize PostgreSQL databases for MLOps system
-- Creates separate databases for MLflow, Airflow, and application

-- Create MLflow database
CREATE DATABASE mlflow;

-- Create Airflow database
CREATE DATABASE airflow;

-- Application databases (mlops_local and mlops_prod) are created by
-- 02_init_app_dbs.sh which runs after this file.  One of them is also
-- created by the POSTGRES_DB env var before any init scripts run.

-- Grant privileges on support databases
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlops_user;
GRANT ALL PRIVILEGES ON DATABASE airflow TO mlops_user;
-- Application DB grants are handled in 02_init_app_dbs.sh
