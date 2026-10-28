# Starts llama-server for one Gemma 4 model. Usage: scripts\serve.ps1 [-Model E2B|E4B] [-Port 8080] [-Ctx 131072] [-Slots 4]
# Note: -Ctx is the WHOLE SERVER's context (`-c`), and llama-server gives each slot -Ctx / -Slots. To give
# every one of N concurrent runs a full 32768-token window, launch -Ctx (32768*N) -Slots N. If the KV does
# not fit, that division is what the server silently keeps: check the banner's kv_unified field (it is not
# exposed over HTTP, and /props reports 32768 either way) or let llm.verify_run_context prove it.
# Note: llama-server writes its banner to stderr; we use Start-Process redirection instead of
# piping so PowerShell's error handling cannot kill the process (that bug cost us one run).
param(
  [ValidateSet('E2B','E4B')][string]$Model = 'E2B',
  [int]$Port = 8080,
  [int]$Ctx = 32768,
  [int]$Slots = 4,
  [string]$GpuLayers = '99',
  [switch]$Sft,
  [string]$Lora = ''
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

if ($Sft -and $Model -eq 'E2B' -and (Test-Path "models\gemma-4-E2B-M4-sft_q4_0-it.gguf")) {
  $gguf = "models\gemma-4-E2B-M4-sft_q4_0-it.gguf"
} elseif ($Sft -and $Model -eq 'E2B' -and (Test-Path "models\gemma-4-E2B-M4-sft-Q8_0.gguf")) {
  $gguf = "models\gemma-4-E2B-M4-sft-Q8_0.gguf"
} else {
  $gguf = "models\gemma-4-$($Model)_q4_0-it.gguf"
}
if (-not (Test-Path $gguf)) { throw "missing $gguf - run scripts\setup_inference.ps1 first" }

if ($Sft -and $Model -eq 'E4B' -and -not $Lora) {
  $Lora = "models\gemma-4-E4B-M4-sft-lora.gguf"
}

$exe = 'tools\llama.cpp\llama-server.exe'
if (-not (Test-Path $exe)) { throw "missing $exe - run scripts\setup_inference.ps1 first" }

$logOut = "logs\server-$Model.log"
$logErr = "logs\server-$Model.err.log"
Write-Host "serving $gguf on http://127.0.0.1:$Port (ctx=$Ctx total, $Slots slot(s) -> $([math]::Floor($Ctx/$Slots)) per run, ngl=$GpuLayers)"
Write-Host "  stdout -> $logOut`n  stderr -> $logErr"

$serverArgs = @(
  '-m', $gguf,
  '-c', $Ctx,
  '-np', $Slots,
  '-ngl', $GpuLayers,
  '--jinja',
  '--flash-attn', 'on',
  '--host', '127.0.0.1',
  '--port', $Port,
  '--metrics'
)
if ($Lora) {
  if (-not (Test-Path $Lora)) { throw "missing LoRA file $Lora" }
  $serverArgs += @('--lora', $Lora)
  Write-Host "  lora   -> $Lora"
}

$p = Start-Process -FilePath $exe -ArgumentList $serverArgs -NoNewWindow -PassThru `
  -RedirectStandardOutput $logOut -RedirectStandardError $logErr

Write-Host "pid=$($p.Id); waiting for exit (Ctrl+C to stop)"
$p.WaitForExit()
Write-Host "server exited with code $($p.ExitCode)"
