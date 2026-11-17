[CmdletBinding()]
param(
    [switch]$Background,
    [switch]$PlanOnly,
    [int[]]$Seeds = @(42),
    [double]$AlignmentWeight = 1.0,
    [int]$TargetLabelsPerClass = 0,
    [string]$Python = $env:PYTHON,
    [string]$DataRoot = $env:JMDA_DATA_ROOT,
    [string]$DataDir
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if (-not $Python) { $Python = 'D:\Anaconda\envs\pytorch\python.exe' }
if (-not $DataRoot) { $DataRoot = 'D:\Code\JMDA-Net\Data' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) { throw "Dataset missing: $DataRoot" }
if (-not $DataDir) {
    $Candidates = @(Get-ChildItem -LiteralPath (Join-Path $Root 'runs/weather_all') -Directory -Filter 'local_*' |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'lists/guided/holdout_protocol_manifest.json') } |
        Sort-Object LastWriteTime -Descending)
    if (-not $Candidates.Count) { throw 'Pass -DataDir with the existing locked guided split directory' }
    $DataDir = Join-Path $Candidates[0].FullName 'lists/guided'
}
$DataDir = (Resolve-Path -LiteralPath $DataDir).Path
if (-not (Test-Path -LiteralPath (Join-Path $DataDir 'holdout_protocol_manifest.json'))) { throw 'Locked split manifest missing' }
if ($Seeds.Count -ne @($Seeds | Select-Object -Unique).Count) { throw 'Repeated seeds' }
if ([double]::IsNaN($AlignmentWeight) -or [double]::IsInfinity($AlignmentWeight) -or $AlignmentWeight -lt 0) { throw 'AlignmentWeight must be finite and nonnegative' }
if ($TargetLabelsPerClass -lt 0 -or ($TargetLabelsPerClass -gt 0 -and $AlignmentWeight -eq 0)) { throw 'A positive label budget requires positive alignment weight' }
$env:JMDA_DATA_ROOT = $DataRoot
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:JMDA_DETERMINISTIC = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$WeightText = $AlignmentWeight.ToString('G', [Globalization.CultureInfo]::InvariantCulture)
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$LogDir = Join-Path $Root 'server_logs/foggy_alignment_launcher'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$LatestPid = Join-Path $LogDir 'latest.pid'
if (-not $PlanOnly -and (Test-Path -LiteralPath $LatestPid)) {
    $PreviousPid = [int](Get-Content -LiteralPath $LatestPid -Raw).Trim()
    if (Get-Process -Id $PreviousPid -ErrorAction SilentlyContinue) {
        throw "Previous foggy alignment queue is still running: PID=$PreviousPid"
    }
}
$Prefix = if ($PlanOnly) { 'plan_' } else { '' }
$BudgetTag = if ($TargetLabelsPerClass -gt 0) { "_${TargetLabelsPerClass}shot" } else { '' }
$RunDir = Join-Path $Root "runs/weather_label_guided/${Prefix}foggy_alignment$($WeightText.Replace('.', 'p'))${BudgetTag}_$Stamp"
$RunArgs = @('-B', '-u', '-X', 'utf8', 'tools/run_label_guided_matrix.py', '--local',
    '--local-lease-timeout-seconds', '0', '--data-dir', $DataDir, '--output', $RunDir,
    '--pairs', 'dann_foggy', 'deepcoral_foggy', '--seeds')
$RunArgs += @($Seeds | ForEach-Object { [string]$_ })
$RunArgs += @('--matched-alignment-weight', $WeightText, '--complete-budget',
    '--parameter-profile', 'reference', '--evaluation-mode', 'validation_ce')
if ($TargetLabelsPerClass -gt 0) {
    $RunArgs += @('--target-labels-per-class', [string]$TargetLabelsPerClass, '--early-stopping-patience', '0')
}
if ($PlanOnly) {
    & $Python @RunArgs
    if ($LASTEXITCODE -ne 0) { throw "Plan failed: exit=$LASTEXITCODE" }
    return
}
& $Python -B -u -X utf8 tools/run_weather_all.py --local --targets foggy --data-root $DataRoot --check-environment
if ($LASTEXITCODE -ne 0) { throw "Environment check failed: exit=$LASTEXITCODE" }
$RunArgs += '--execute'
Write-Host "Foggy: DANN + Deep CORAL; seeds=$($Seeds -join ','); alignment weight=$WeightText"
$Protocol = if ($AlignmentWeight -eq 0) { 'LABEL_GUIDED_CLASS_MATCHING' } else { 'LABEL_GUIDED_CLASS_ALIGNMENT' }
if ($TargetLabelsPerClass -gt 0) { $Protocol = 'FEWSHOT_TARGET_CLASS_ALIGNMENT'; Write-Host "Fixed labels per class=$TargetLabelsPerClass; all target images remain in native DA; fixed 40 epochs, early stopping off." }
Write-Host "Protocol=$Protocol; reference learning rates; validation-CE selection; no heldout evaluation."
Write-Host "Existing local training holds priority; this queue waits for the shared training lease."
Write-Host "Run directory=$RunDir"
if ($Background) {
    $Stdout = Join-Path $LogDir "$Stamp.out.log"
    $Stderr = Join-Path $LogDir "$Stamp.err.log"
    $ArgumentString = & $Python -X utf8 -c 'import subprocess,sys; print(subprocess.list2cmdline(sys.argv[1:]))' @RunArgs
    if ($LASTEXITCODE -ne 0) { throw 'Failed to quote Python arguments' }
    $WorkerProcess = Start-Process -FilePath $Python -ArgumentList $ArgumentString -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
    Set-Content -LiteralPath $LatestPid -Value $WorkerProcess.Id -Encoding ascii
    Set-Content -LiteralPath (Join-Path $LogDir "$Stamp.pid") -Value $WorkerProcess.Id -Encoding ascii
    Set-Content -LiteralPath (Join-Path $LogDir 'latest.run.txt') -Value $RunDir -Encoding utf8
    Write-Host "PID=$($WorkerProcess.Id)"
    Write-Host "stdout=$Stdout"
    Write-Host "stderr=$Stderr"
    Write-Host "Monitor: Get-Content -Wait -LiteralPath '$Stdout'"
} else {
    & $Python @RunArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed: exit=$LASTEXITCODE; see $RunDir" }
}
