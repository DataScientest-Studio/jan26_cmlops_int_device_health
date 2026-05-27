<#
.SYNOPSIS
    Start background kubectl port-forward jobs for all MLOps services.

.DESCRIPTION
    Launches one port-forward background job per service so that every
    component is reachable on localhost after "make k8s-up":

        Service       Local port   Container port
        ─────────     ──────────   ──────────────
        api           8000         8000
        streamlit     8501         8501
        mlflow        5000         5000
        airflow       8080         8080
        grafana       3000         3000
        prometheus    9090         9090
        nginx         8888         80

    Job names are prefixed with "mlops-pf-" so they can be stopped cleanly
    with k8s_stop_ports.ps1 or "make k8s-ports-stop".

.EXAMPLE
    .\scripts\k8s_port_forward.ps1
    .\scripts\k8s_port_forward.ps1 -Namespace mlops -Wait
#>

param(
    [string]$Namespace = "mlops",
    [switch]$Wait                  # Block until Ctrl-C (useful in a dedicated terminal)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Service map: name → "local:remote" ────────────────────────────────────────
$services = [ordered]@{
    "api"        = "8000:8000"
    "streamlit"  = "8502:8501"
    "mlflow"     = "5000:5000"
    "postgres"   = "5434:5432"
    "airflow"    = "8080:8080"
    "grafana"    = "3000:3000"
    "prometheus" = "9090:9090"
    "nginx"      = "8888:80"
}

# ── Verify kubectl is available ────────────────────────────────────────────────
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Write-Error "kubectl not found on PATH. Install kubectl and ensure Docker Desktop K8s is enabled."
    exit 1
}

# ── Verify namespace exists ────────────────────────────────────────────────────
$nsCheck = kubectl get namespace $Namespace 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Namespace '$Namespace' does not exist.`nRun 'make k8s-up' first to deploy the stack."
    exit 1
}

Write-Host ""
Write-Host "Starting port-forward jobs for namespace '$Namespace' ..." -ForegroundColor Cyan
Write-Host ""

$jobs = @()
foreach ($svc in $services.Keys) {
    $ports    = $services[$svc]
    $jobName  = "mlops-pf-$svc"

    # Remove any stale job with the same name
    Get-Job -Name $jobName -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue

    $job = Start-Job -Name $jobName -ScriptBlock {
        param($ns, $svcName, $portMap)
        while ($true) {
            kubectl port-forward "svc/$svcName" $portMap -n $ns 2>&1
            Start-Sleep -Seconds 2  # brief pause before reconnect on disconnect
        }
    } -ArgumentList $Namespace, $svc, $ports

    $jobs += $job
    Write-Host ("  {0,-12} localhost:{1,-5}  -> {2}/{3}" -f $svc, ($ports -split ':')[0], $Namespace, $svc) -ForegroundColor Green
}

Write-Host ""
Write-Host "All port-forward jobs started." -ForegroundColor Cyan
Write-Host "  Stop with:  make k8s-ports-stop   or   .\scripts\k8s_stop_ports.ps1"
Write-Host ""
Write-Host "  Open services:"
Write-Host "    Streamlit UI : http://localhost:8502  (K8s pod; 'make ui' uses 8501)"
Write-Host "    API          : http://localhost:8000"
Write-Host "    MLflow       : http://localhost:5000"
Write-Host "    Airflow      : http://localhost:8080"
Write-Host "    Grafana      : http://localhost:3000"
Write-Host "    Prometheus   : http://localhost:9090"
Write-Host "    Nginx (entry): http://localhost:8888"
Write-Host "    PostgreSQL   : localhost:5434 (for host Streamlit / make ui)"
Write-Host ""

if ($Wait) {
    Write-Host "Waiting (Ctrl-C to stop all port-forwards)..." -ForegroundColor Yellow
    try {
        while ($true) { Start-Sleep -Seconds 5 }
    }
    finally {
        Write-Host "Stopping port-forward jobs..." -ForegroundColor Yellow
        $jobs | Stop-Job -ErrorAction SilentlyContinue
        $jobs | Remove-Job -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped." -ForegroundColor Green
    }
}
