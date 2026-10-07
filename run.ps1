$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot '.venv312\Scripts\python.exe'
$runtimePython = $venvPython

if (-not (Test-Path $runtimePython)) {
    $runtimePython = 'C:\Users\Alexander\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
}

if (-not (Test-Path $runtimePython)) {
    throw 'No se encontró el entorno ejecutable de Python.'
}

# Preferimos el Python local de Windows porque SAPI5 requiere acceso a la
# sesión interactiva donde está instalada la voz. Solo el runtime de reserva
# usa las dependencias empaquetadas por Codex.
if ($runtimePython -notlike $venvPython) {
    $packageRoot = Join-Path $projectRoot '.python-packages'
    if (-not (Test-Path $packageRoot)) {
        throw 'Faltan las dependencias del proyecto. Ejecuta el comando de instalación indicado en README.md.'
    }
    $env:PYTHONPATH = $packageRoot
} else {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
}

Set-Location $projectRoot
Write-Host "Iniciando aplicacion grafica..." -ForegroundColor Cyan
& $runtimePython main.py



