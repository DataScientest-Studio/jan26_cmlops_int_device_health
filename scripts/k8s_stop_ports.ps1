<#
.SYNOPSIS
    Stop all running kubectl port-forward background jobs for MLOps services.

.EXAMPLE
    .\scripts\k8s_stop_ports.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "SilentlyContinue"

$jobs = Get-Job -Name "mlops-pf-*" -ErrorAction SilentlyContinue

if ($null -eq $jobs -or $jobs.Count -eq 0) {
    Write-Host "No MLOps port-forward jobs are running." -ForegroundColor Yellow
    exit 0
}

Write-Host "Stopping port-forward jobs..." -ForegroundColor Cyan
foreach ($job in $jobs) {
    Stop-Job  -Id $job.Id -ErrorAction SilentlyContinue
    Remove-Job -Id $job.Id -Force -ErrorAction SilentlyContinue
    Write-Host ("  Stopped: {0}" -f $job.Name) -ForegroundColor Green
}
Write-Host "All MLOps port-forward jobs stopped." -ForegroundColor Green
