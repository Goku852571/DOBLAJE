$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot '.venv312\Scripts\python.exe'
$runtimePython = $venvPython

if (-not (Test-Path -LiteralPath $runtimePython)) {
    $runtimePython = Join-Path $projectRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $runtimePython)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw 'Instala Python 3.12 y las dependencias según INTEGRACION_TIKTOK.md.'
    }
    $runtimePython = $pythonCommand.Source
    $packageRoot = Join-Path $projectRoot '.python-packages'
    if (Test-Path -LiteralPath $packageRoot) {
        $env:PYTHONPATH = $packageRoot
    }
} else {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
}
Set-Location -LiteralPath $projectRoot
Write-Host 'Iniciando aplicacion grafica...' -ForegroundColor Cyan
& $runtimePython (Join-Path $projectRoot 'main.py')
