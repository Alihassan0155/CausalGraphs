$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Write-Host 'Creating .venv with the current Python interpreter...'
    py -3.12 -m venv .venv
}

$Python = Join-Path $Root '.venv\Scripts\python.exe'
& $Python -m pip install --upgrade 'pip==24.2' 'setuptools==75.1.0' 'wheel==0.44.0'
& $Python -m pip install --no-cache-dir -r requirements.txt

foreach ($Folder in @('data', 'processed', 'graphs', 'results')) {
    New-Item -ItemType Directory -Force -Path $Folder | Out-Null
}

& $Python (Join-Path $PSScriptRoot 'verify_setup.py')
