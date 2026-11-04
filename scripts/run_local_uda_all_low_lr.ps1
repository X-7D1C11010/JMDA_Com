param(
    [string]$Target = 'backlight',
    [int]$Seed = 42,
    [int]$BatchSize = 16,
    [int]$CdanIterations = 2000,
    [int]$Epochs = 0,
    [string]$Python = 'D:\Anaconda\envs\pytorch\python.exe',
    [string]$OutDir = '',
    [string]$DataDir = '',
    [double]$DANNLearningRate = 0.0005,
    [double]$DeepCoralLearningRate = 0.0005,
    [double]$CDANLearningRate = 0.0005,
    [double]$MDDLearningRate = 0.002,
    [double]$MCCLearningRate = 0.0025
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$validTargets = @('backlight', 'rainy', 'foggy', 'dark')
if ($Target -notin $validTargets) {
    throw "Unknown target '$Target'. Valid targets: $($validTargets -join ', ')"
}
if ($BatchSize -lt 2) { throw 'BatchSize must be at least 2.' }
if ($CdanIterations -le 0) { throw 'CdanIterations must be positive.' }
if ($Epochs -lt 0) { throw 'Epochs must be zero (method defaults) or positive.' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python interpreter not found: $Python"
}
if (-not $OutDir) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutDir = Join-Path $Repo "efficiency_results\uda_local_all_low_lr_${Target}_seed${Seed}_$stamp"
}

$arguments = @{
    Seeds = @($Seed)
    Targets = @($Target)
    IncludeMethods = @('DANN', 'DeepCORAL', 'CDAN', 'MDD', 'MCC')
    IncludePausedMethods = $true
    Python = $Python
    OutDir = $OutDir
    CdanIterations = $CdanIterations
    ImageBatchSize = $BatchSize
    Epochs = $Epochs
    DANNLearningRate = $DANNLearningRate
    DeepCoralLearningRate = $DeepCoralLearningRate
    CDANLearningRate = $CDANLearningRate
    MDDLearningRate = $MDDLearningRate
    MCCLearningRate = $MCCLearningRate
    SkipBenchmark = $true
}
if ($DataDir) { $arguments.DataDir = $DataDir }

Write-Host 'Local sequential low-LR run: DANN, DeepCORAL, CDAN, MDD, MCC.'
Write-Host "Target=$Target Seed=$Seed BatchSize=$BatchSize CDAN iterations=$CdanIterations"
Write-Host ("LR: DANN=$DANNLearningRate, DeepCORAL=$DeepCoralLearningRate, " +
           "CDAN=$CDANLearningRate, MDD=$MDDLearningRate, " +
           "MCC=$MCCLearningRate")
& (Join-Path $PSScriptRoot 'run_uda_multimodal.ps1') @arguments
exit $LASTEXITCODE
