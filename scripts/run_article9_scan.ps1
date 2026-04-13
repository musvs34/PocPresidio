Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venvPresidio\Scripts\python.exe"
$configPath = Join-Path $repoRoot "configs\article9_categories.yml"
$inputDir = Join-Path $repoRoot "data\raw"
$outputDir = Join-Path $repoRoot "data\processed"
$scriptPath = Join-Path $repoRoot "src\run_article9_scan.py"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python venv introuvable: $pythonExe"
}

& $pythonExe $scriptPath --input-dir $inputDir --config $configPath --output-dir $outputDir
