$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$RunRoot = Join-Path $Root "results\sccvae_baseline\runs\fresh_$Stamp"
New-Item -ItemType Directory -Path $RunRoot | Out-Null
$LogRoot = Join-Path $RunRoot 'logs'
$ProvenanceRoot = Join-Path $RunRoot 'provenance'
New-Item -ItemType Directory -Path $LogRoot, $ProvenanceRoot | Out-Null
$env:WANDB_MODE = 'offline'
& $Python -m pip freeze | Out-File "$ProvenanceRoot\packages.txt" -Encoding utf8
git rev-parse HEAD | Out-File "$ProvenanceRoot\git_commit.txt" -Encoding utf8
git diff | Out-File "$ProvenanceRoot\source_diff.txt" -Encoding utf8
nvidia-smi | Out-File "$ProvenanceRoot\gpu.txt" -Encoding utf8
Write-Host "Run directory: $RunRoot"
# Windows PowerShell treats native stderr as ErrorRecords. W&B and tqdm
# legitimately write there; only the Python exit code indicates failure.
$PreviousErrorPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $Python -u (Join-Path $PSScriptRoot 'run_baseline_checkpointed.py') -o $RunRoot -x ood_split_0_full -d 0 --ood --split-num 0 --seed 20 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "$LogRoot\training.log"
    $TrainingExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PreviousErrorPreference
}
$TrainingExitCode | Out-File "$ProvenanceRoot\exit_code.txt" -Encoding utf8
if ($TrainingExitCode -ne 0) { throw "Training failed (exit $TrainingExitCode): inspect $LogRoot\training.log" }
