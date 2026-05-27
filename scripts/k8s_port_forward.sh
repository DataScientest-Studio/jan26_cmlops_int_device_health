#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# k8s_port_forward.sh — start background kubectl port-forward for all services
#
# Usage:
#   bash scripts/k8s_port_forward.sh         # start in background
#   bash scripts/k8s_port_forward.sh --wait  # block until Ctrl-C
#   bash scripts/k8s_port_forward.sh --stop  # kill running forwards
#
# Port map:
#   api        8000:8000
#   streamlit  8502:8501  (K8s pod; 'make ui' uses 8501)
#   mlflow     5000:5000
#   postgres   5434:5432  (avoids conflict with local PG on 5432 / Docker stack on 5433)
#   airflow    8080:8080
#   grafana    3000:3000
#   prometheus 9090:9090
#   nginx      8888:80
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NAMESPACE="${K8S_NAMESPACE:-mlops}"
PID_FILE="/tmp/mlops_port_forward_pids"
WAIT=false
STOP=false

for arg in "$@"; do
    case "$arg" in
        --wait) WAIT=true ;;
        --stop) STOP=true ;;
    esac
done

# ── Shared cleanup helper ──────────────────────────────────────────────────────
# Kills tracked subshell PIDs AND any stray kubectl port-forward processes.
# This handles both clean shutdowns and the case where subshells were killed
# but their kubectl children became orphans still holding the ports.
_cleanup() {
    if [[ -f "$PID_FILE" ]]; then
        while IFS= read -r pid; do
            kill -- "-${pid}" 2>/dev/null || kill "$pid" 2>/dev/null || true
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    # Kill any remaining kubectl port-forward processes for this namespace.
    # Works on Git Bash (Windows), macOS, and Linux.
    pkill -f "kubectl port-forward svc/" 2>/dev/null || true
    sleep 1
}

# ── Stop mode ─────────────────────────────────────────────────────────────────
if $STOP; then
    echo "Stopping port-forward processes..."
    _cleanup
    echo "Done."
    exit 0
fi

# ── Verify kubectl ─────────────────────────────────────────────────────────────
if ! command -v kubectl &>/dev/null; then
    echo "ERROR: kubectl not found on PATH." >&2
    exit 1
fi

if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "ERROR: Namespace '$NAMESPACE' not found. Run 'make k8s-up' first." >&2
    exit 1
fi

# ── Start port-forwards ────────────────────────────────────────────────────────
# Note: bash 3.2 (macOS default) does not support declare -A.
# Use a plain array of "svc=local:remote" entries instead.
services=(
    "api=8000:8000"
    "streamlit=8502:8501"
    "mlflow=5000:5000"
    "postgres=5434:5432"
    "airflow=8080:8080"
    "grafana=3000:3000"
    "prometheus=9090:9090"
    "nginx=8888:80"
)

echo ""
echo "Starting port-forward jobs for namespace '$NAMESPACE' ..."
echo ""

# Kill any existing port-forwards BEFORE starting new ones.
# This prevents "address already in use" errors when re-running the script
# without an explicit --stop first.
_cleanup

for entry in "${services[@]}"; do
    svc="${entry%%=*}"
    ports="${entry#*=}"
    local_port="${ports%%:*}"
    # Reconnect loop in background.
    # nohup prevents SIGHUP when the parent script (or make) exits.
    # || true prevents set -e from exiting the subshell when kubectl disconnects.
    # /dev/null redirects suppress the nohup.out file.
    _svc="$svc" _ports="$ports" _ns="$NAMESPACE" \
    nohup bash -c '
        while true; do
            kubectl port-forward "svc/$_svc" "$_ports" -n "$_ns" 2>/dev/null || true
            sleep 2
        done
    ' >/dev/null 2>&1 &
    pid=$!
    disown "$pid"
    echo "$pid" >> "$PID_FILE"
    printf "  %-12s localhost:%-5s -> %s/%s\n" "$svc" "$local_port" "$NAMESPACE" "$svc"
done

echo ""
echo "All port-forward jobs started. PIDs saved to $PID_FILE"
echo "  Stop with: bash scripts/k8s_port_forward.sh --stop"
echo "          or: make k8s-ports-stop"
echo ""
echo "  Open services:"
echo "    Streamlit UI : http://localhost:8502  (K8s pod; 'make ui' uses 8501)"
echo "    API          : http://localhost:8000"
echo "    MLflow       : http://localhost:5000"
echo "    Airflow      : http://localhost:8080"
echo "    Grafana      : http://localhost:3000"
echo "    Prometheus   : http://localhost:9090"
echo "    Nginx (entry): http://localhost:8888"
echo "    PostgreSQL   : localhost:5434 (for host Streamlit / make ui)"
echo ""

if $WAIT; then
    echo "Waiting (Ctrl-C to stop all port-forwards)..."
    trap '_cleanup; exit 0' INT TERM
    while true; do sleep 5; done
fi
