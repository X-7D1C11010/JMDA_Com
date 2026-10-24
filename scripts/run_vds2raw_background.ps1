param(
    [ValidateSet('formal', 'smoke')]
    [string]$Mode = 'formal'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$LauncherDir = Join-Path $RepoRoot 'server_logs\vds2raw_local_launcher'
New-Item -ItemType Directory -Force -Path $LauncherDir | Out-Null
$stamp = '{0}_{1}_{2}' -f $Mode, (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID
$stdoutLog = Join-Path $LauncherDir "$stamp.out.log"
$stderrLog = Join-Path $LauncherDir "$stamp.err.log"
$pidFile = Join-Path $LauncherDir "$stamp.pid"
$workerScript = Join-Path $PSScriptRoot 'run_vds2raw_methods.ps1'
$pwsh = Join-Path $PSHOME 'pwsh.exe'
if (-not (Test-Path -LiteralPath $pwsh -PathType Leaf)) {
    $pwsh = Join-Path $PSHOME 'powershell.exe'
}

$process = Start-Process `
    -FilePath $pwsh `
    -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', $workerScript, '-Mode', $Mode
    ) `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii

Write-Host 'VDS2Raw local background run started.'
Write-Host "PID=$($process.Id)"
Write-Host "stdout=$stdoutLog"
Write-Host "stderr=$stderrLog"
Write-Host "PID file=$pidFile"
Write-Host "Monitor: Get-Content -Wait -LiteralPath '$stdoutLog'"

