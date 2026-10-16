# Downloads the llama.cpp Windows CUDA binaries and the Gemma 4 QAT GGUF weights.
# Idempotent: skips anything already present. Run from the repo root.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$Build = 'b11696'
$LlamaDir = 'tools\llama.cpp'
$ModelDir = 'models'
New-Item -ItemType Directory -Force -Path $LlamaDir, $ModelDir, 'tools\downloads' | Out-Null

# --- 1. llama.cpp binaries (CUDA 13.4 x64) + bundled CUDA runtime -------------
$assets = @(
  "llama-$Build-bin-win-cuda-13.4-x64.zip",
  "cudart-llama-bin-win-cuda-13.4-x64.zip"
)
foreach ($a in $assets) {
  $zip = "tools\downloads\$a"
  if (-not (Test-Path "$LlamaDir\llama-server.exe") -or -not (Test-Path $zip)) {
    if (-not (Test-Path $zip)) {
      Write-Host "[llama.cpp] downloading $a"
      curl.exe -L --fail --retry 3 -o $zip `
        "https://github.com/ggml-org/llama.cpp/releases/download/$Build/$a"
    }
    Write-Host "[llama.cpp] extracting $a"
    Expand-Archive -Path $zip -DestinationPath $LlamaDir -Force
  }
}

# --- 2. Gemma 4 QAT GGUFs (official Google quantisation-aware trained Q4_0) ---
$models = @{
  'gemma-4-E2B_q4_0-it.gguf' = 'google/gemma-4-E2B-it-qat-q4_0-gguf'
  'gemma-4-E4B_q4_0-it.gguf' = 'google/gemma-4-E4B-it-qat-q4_0-gguf'
}
foreach ($f in $models.Keys) {
  if (Test-Path "$ModelDir\$f") {
    Write-Host "[models] $f already present, skipping"
    continue
  }
  Write-Host "[models] downloading $f from $($models[$f])"
  .\.venv\Scripts\hf.exe download $models[$f] $f --local-dir $ModelDir
}

Write-Host "`n[verify]"
& "$LlamaDir\llama-server.exe" --version
Get-ChildItem $ModelDir | Select-Object Name, @{n='GB';e={[math]::Round($_.Length/1GB,2)}}
