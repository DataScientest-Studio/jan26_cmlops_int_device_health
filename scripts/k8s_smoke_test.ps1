<#
.SYNOPSIS
    Smoke-test all MLOps K8s services by hitting their health endpoints.

.DESCRIPTION
    Assumes port-forwards are active (run k8s_port_forward.ps1 first).
    Tests:
        api        http://localhost:8000/
        mlflow     http://localhost:5000/
        airflow    http://localhost:8080/health
        grafana    http://localhost:3000/api/health
        prometheus http://localhost:9090/-/healthy
        streamlit  http://localhost:8501/  (expect 200 or redirect)

    Exits with code 0 if all tests pass, 1 if any fail.

.EXAMPLE
    .\scripts\k8s_smoke_test.ps1
    .\scripts\k8s_smoke_test.ps1 -TimeoutSec 10
#>

param(
    [int]$TimeoutSec = 5,
    [switch]$Quiet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "SilentlyContinue"

# ── Service health endpoints ───────────────────────────────────────────────────
$checks = @(
    @{ Name = "api";        Url = "http://localhost:8000/";             OkCodes = @(200, 307) }
    @{ Name = "streamlit";  Url = "http://localhost:8501/";             OkCodes = @(200, 302) }
    @{ Name = "mlflow";     Url = "http://localhost:5000/";             OkCodes = @(200, 302) }
    @{ Name = "airflow";    Url = "http://localhost:8080/health";       OkCodes = @(200) }
    @{ Name = "grafana";    Url = "http://localhost:3000/api/health";   OkCodes = @(200) }
    @{ Name = "prometheus"; Url = "http://localhost:9090/-/healthy";    OkCodes = @(200) }
)

$passed  = 0
$failed  = 0
$results = @()

Write-Host ""
Write-Host "MLOps K8s Smoke Test" -ForegroundColor Cyan
Write-Host ("=" * 50)
Write-Host ""

foreach ($chk in $checks) {
    $status = $null
    $ok     = $false
    $err    = ""

    try {
        $resp   = Invoke-WebRequest -Uri $chk.Url -TimeoutSec $TimeoutSec -UseBasicParsing -ErrorAction Stop
        $status = $resp.StatusCode
        $ok     = $chk.OkCodes -contains $status
    }
    catch [System.Net.WebException] {
        $status = [int]$_.Exception.Response.StatusCode
        $ok     = $chk.OkCodes -contains $status
        if (-not $ok) { $err = $_.Exception.Message }
    }
    catch {
        $status = 0
        $err    = $_.Exception.Message
    }

    if ($ok) {
        $passed++
        $symbol = "[PASS]"
        $color  = "Green"
    }
    else {
        $failed++
        $symbol = "[FAIL]"
        $color  = "Red"
    }

    $line = "{0}  {1,-12}  HTTP {2}  {3}" -f $symbol, $chk.Name, $status, $chk.Url
    Write-Host $line -ForegroundColor $color
    if ($err -and -not $Quiet) {
        Write-Host ("         Error: {0}" -f $err) -ForegroundColor DarkGray
    }

    $results += [PSCustomObject]@{ Name = $chk.Name; Url = $chk.Url; Status = $status; Pass = $ok; Error = $err }
}

Write-Host ""
Write-Host ("=" * 50)
$summaryColor = if ($failed -eq 0) { "Green" } else { "Red" }
Write-Host ("Result: {0} passed, {1} failed" -f $passed, $failed) -ForegroundColor $summaryColor
Write-Host ""

if ($failed -gt 0) {
    Write-Host "Tip: make sure port-forwards are running:" -ForegroundColor Yellow
    Write-Host "     make k8s-ports" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

exit 0
