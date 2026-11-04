param(
    [string]$Target = 'backlight',
    [int]$Seed = 42,
    [int]$BatchSize = 16,
    [int]$CdanIterations = 2000,
    [string]$Python = 'D:\Anaconda\envs\pytorch\python.exe',
    [string]$OutDir = '',
    [string]$DataDir = ''
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$validTargets = @('backlight', 'rainy', 'foggy', 'dark')
if ($Target -notin $validTargets) {
    throw "Unknown target '$Target'. Valid targets: $($validTargets -join ', ')"
}
if ($BatchSize -lt 2) { throw 'BatchSize must be at least 2.' }
if ($CdanIterations -le 0) { throw 'CdanIterations must be positive.' }
if (-not $OutDir) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutDir = Join-Path $Repo "efficiency_results\uda_local_three_${Target}_seed${Seed}_$stamp"
}

$arguments = @{
    Seeds = @($Seed)
    Targets = @($Target)
    IncludeMethods = @('DANN', 'DeepCORAL', 'CDAN')
    Python = $Python
    OutDir = $OutDir
    CdanIterations = $CdanIterations
    ImageBatchSize = $BatchSize
    SkipBenchmark = $true
}
if ($DataDir) { $arguments.DataDir = $DataDir }

Write-Host 'Local split: DANN, DeepCORAL, CDAN only.'
Write-Host "Target=$Target Seed=$Seed BatchSize=$BatchSize CDAN iterations=$CdanIterations"
& (Join-Path $PSScriptRoot 'run_uda_multimodal.ps1') @arguments
exit $LASTEXITCODE
