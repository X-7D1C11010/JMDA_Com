param(
    [string]$Target = 'backlight',
    [int]$Seed = 42,
    [int]$BatchSize = 16,
    [string[]]$IncludeMethods = @('DANN', 'DeepCORAL', 'CDAN'),
    [string]$Python = 'D:\Anaconda\envs\pytorch\python.exe',
    [string]$DataDir = ''
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$LauncherDir = Join-Path $RepoRoot 'server_logs\uda_tuned_local_launcher'
New-Item -ItemType Directory -Force -Path $LauncherDir | Out-Null

$stamp = '{0}_seed{1}_{2}_{3}' -f `
    $Target, $Seed, (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID
$stdoutLog = Join-Path $LauncherDir "$stamp.out.log"
$stderrLog = Join-Path $LauncherDir "$stamp.err.log"
$pidFile = Join-Path $LauncherDir "$stamp.pid"
$workerScript = Join-Path $PSScriptRoot 'run_uda_tuned.ps1'
$pwsh = Join-Path $PSHOME 'pwsh.exe'
if (-not (Test-Path -LiteralPath $pwsh -PathType Leaf)) {
    $pwsh = Join-Path $PSHOME 'powershell.exe'
}

$arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', $workerScript,
    '-Target', $Target,
    '-Seed', $Seed,
    '-BatchSize', $BatchSize,
    '-IncludeMethods', ($IncludeMethods -join ','),
    '-Python', $Python
)
if ($DataDir) { $arguments += @('-DataDir', $DataDir) }

$process = Start-Process `
    -FilePath $pwsh `
    -ArgumentList $arguments `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii

Write-Host 'Tuned local UDA run started in the background.'
Write-Host "PID=$($process.Id)"
Write-Host "stdout=$stdoutLog"
Write-Host "stderr=$stderrLog"
Write-Host "PID file=$pidFile"
Write-Host "Monitor: Get-Content -Wait -LiteralPath '$stdoutLog'"
