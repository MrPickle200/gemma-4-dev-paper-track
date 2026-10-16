# Starts llama-server for one Gemma 4 model. Usage: scripts\serve.ps1 [-Model E2B|E4B] [-Port 8080] [-Ctx 32768]
# Note: llama-server writes its banner to stderr; we use Start-Process redirection instead of
# piping so PowerShell's error handling cannot kill the process (that bug cost us one run).
param(
  [ValidateSet('E2B','E4B')][string]$Model = 'E2B',
  [int]$Port = 8080,
  [int]$Ctx = 32768,
  [string]$GpuLayers = '99'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$gguf = "models\gemma-4-$($Model)_q4_0-it.gguf"
if (-not (Test-Path $gguf)) { throw "missing $gguf - run scripts\setup_inference.ps1 first" }

$exe = 'tools\llama.cpp\llama-server.exe'
if (-not (Test-Path $exe)) { throw "missing $exe - run scripts\setup_inference.ps1 first" }

$logOut = "logs\server-$Model.log"
$logErr = "logs\server-$Model.err.log"
Write-Host "serving $gguf on http://127.0.0.1:$Port (ctx=$Ctx, ngl=$GpuLayers)"
Write-Host "  stdout -> $logOut`n  stderr -> $logErr"

$serverArgs = @(
  '-m', $gguf,
  '-c', $Ctx,
  '-ngl', $GpuLayers,
  '--jinja',
  '--flash-attn', 'on',
  '--host', '127.0.0.1',
  '--port', $Port,
  '--metrics'
)

$p = Start-Process -FilePath $exe -ArgumentList $serverArgs -NoNewWindow -PassThru `
  -RedirectStandardOutput $logOut -RedirectStandardError $logErr

Write-Host "pid=$($p.Id); waiting for exit (Ctrl+C to stop)"
$p.WaitForExit()
Write-Host "server exited with code $($p.ExitCode)"
