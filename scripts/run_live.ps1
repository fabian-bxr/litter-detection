<#
.SYNOPSIS
    Startet die Live-YOLO-Litter-Detection auf dem Zenoh-Kamerastream.

.DESCRIPTION
    Prüft zuerst, ob der Zenoh-Router erreichbar ist, startet dann die
    Kamera-Bridge (publiziert camera/frame) in einem eigenen Fenster und
    danach den YOLO-Detector im aktuellen Fenster.

.EXAMPLE
    ./scripts/run_live.ps1
    ./scripts/run_live.ps1 -Source webcam -WebcamId 0
    ./scripts/run_live.ps1 -Conf 0.25 -Dilate 0          # weniger Fehlalarme
    ./scripts/run_live.ps1 -Endpoint tcp/127.0.0.1:7447  # lokaler Router
#>
param(
    [ValidateSet("go2", "webcam")]
    [string]$Source   = "go2",
    [int]   $WebcamId = 0,
    [string]$Endpoint = $env:ZENOH_ROUTER_ENDPOINT,
    [string]$Conf,                 # optional: YOLO_CONF
    [string]$ImgSz,                # optional: YOLO_IMGSZ
    [string]$Dilate,               # optional: YOLO_MASK_DILATE
    [switch]$Force                 # Erreichbarkeits-Check überspringen
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot   # Repo-Root (Skript liegt in scripts/)

if (-not $Endpoint) { $Endpoint = "tcp/192.168.4.207:7447" }
$env:ZENOH_ROUTER_ENDPOINT = $Endpoint

# Host:Port aus "tcp/HOST:PORT" parsen
$hostPort = $Endpoint -replace '^tcp/', ''
$zhost = $hostPort.Split(':')[0]
$zport = [int]$hostPort.Split(':')[1]

Write-Host "Zenoh-Router: $Endpoint"
if (-not $Force) {
    Write-Host "Pruefe Erreichbarkeit von $zhost`:$zport ..." -NoNewline
    $ok = Test-NetConnection -ComputerName $zhost -Port $zport -InformationLevel Quiet -WarningAction SilentlyContinue
    if (-not $ok) {
        Write-Host " NICHT erreichbar." -ForegroundColor Red
        Write-Host "Optionen:"
        Write-Host "  - Go2/Router pruefen"
        Write-Host "  - lokalen Router: docker run --rm -p 7447:7447 eclipse/zenoh:latest"
        Write-Host "    dann: ./scripts/run_live.ps1 -Endpoint tcp/127.0.0.1:7447"
        Write-Host "  - Check ueberspringen: -Force"
        exit 1
    }
    Write-Host " OK." -ForegroundColor Green
}

# Optionale YOLO-Tuning-Overrides (vom Detector im aktuellen Fenster geerbt)
if ($Conf)   { $env:YOLO_CONF = $Conf }
if ($ImgSz)  { $env:YOLO_IMGSZ = $ImgSz }
if ($Dilate) { $env:YOLO_MASK_DILATE = $Dilate }

# Kamera-Argumente
$camArgs = "run camera --source $Source"
if ($Source -eq "webcam") { $camArgs += " --id $WebcamId" }

Write-Host "Starte Kamera ($Source) in neuem Fenster ..."
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd `"$repo`"; `$env:ZENOH_ROUTER_ENDPOINT='$Endpoint'; uv $camArgs"
)

Start-Sleep -Seconds 2
Write-Host "Starte YOLO-Detector (Strg+C zum Beenden) ..." -ForegroundColor Cyan
Set-Location $repo
uv run detector
