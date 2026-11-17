[CmdletBinding()]
param(
    [switch]$Background,
    [switch]$PlanOnly,
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [string]$Python = $env:PYTHON,
    [string]$DataRoot = $env:JMDA_DATA_ROOT
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if (-not $Python) { $Python = 'D:\Anaconda\envs\pytorch\python.exe' }
if (-not $DataRoot) { $DataRoot = 'D:\Code\JMDA-Net\Data' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) { throw "Dataset missing: $DataRoot" }
$env:JMDA_DATA_ROOT = $DataRoot
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:JMDA_DETERMINISTIC = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Prefix = if ($PlanOnly) { 'plan_local_fewshot_' } else { 'local_fewshot_' }
$RunDir = Join-Path $Root "runs/weather_label_guided/${Prefix}$Stamp"
$RunArgs = @('-B', '-u', '-X', 'utf8', 'tools/run_weather_fewshot_suite.py', '--machine', 'local', '--seeds')
$RunArgs += @($Seeds | ForEach-Object { [string]$_ })
$RunArgs += @('--data-root', $DataRoot, '--output', $RunDir)
if ($PlanOnly) {
    & $Python @RunArgs
    if ($LASTEXITCODE -ne 0) { throw "Plan failed: exit=$LASTEXITCODE" }
    return
}
$LogDir = Join-Path $Root 'server_logs/weather_fewshot_local_launcher'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$LatestPid = Join-Path $LogDir 'latest.pid'
if (Test-Path -LiteralPath $LatestPid) {
    $PreviousPid = [int](Get-Content -LiteralPath $LatestPid -Raw).Trim()
    if (Get-Process -Id $PreviousPid -ErrorAction SilentlyContinue) {
        throw "Previous local few-shot queue is still running: PID=$PreviousPid"
    }
}
& $Python -B -u -X utf8 tools/run_weather_fewshot_suite.py --machine local --seeds $Seeds --data-root $DataRoot --check-environment
if ($LASTEXITCODE -ne 0) { throw "Environment check failed: exit=$LASTEXITCODE" }
$RunArgs += '--execute'
Write-Host 'Local: rainy five methods; foggy DANN/DeepCORAL/CDAN.'
Write-Host "Seeds=$($Seeds -join ','); jobs=$($Seeds.Count * 8); 40 epochs, early stopping off."
Write-Host 'Protocol=FEWSHOT_TARGET_CLASS_ALIGNMENT; 3 labels/class; alignment=0.1; teacher losses off.'
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
