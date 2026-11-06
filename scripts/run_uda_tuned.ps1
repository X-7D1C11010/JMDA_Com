param(
    [string]$Target = 'backlight',
    [int]$Seed = 42,
    [int]$BatchSize = 16,
    [string[]]$IncludeMethods = @('DANN', 'DeepCORAL', 'CDAN', 'MDD', 'MCC'),
    [string]$Python = 'D:\Anaconda\envs\pytorch\python.exe',
    [string]$OutDir = '',
    [string]$DataDir = ''
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$IncludeMethods = @(
    $IncludeMethods |
        ForEach-Object { $_ -split ',' } |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
$validTargets = @('backlight', 'rainy', 'foggy', 'dark')
$validMethods = @('DANN', 'DeepCORAL', 'CDAN', 'MDD', 'MCC')
if ($Target -notin $validTargets) {
    throw "Unknown target '$Target'. Valid targets: $($validTargets -join ', ')"
}
foreach ($method in $IncludeMethods) {
    if ($method -notin $validMethods) {
        throw "Unknown method '$method'. Valid methods: $($validMethods -join ', ')"
    }
}
if ($BatchSize -lt 2) { throw 'BatchSize must be at least 2.' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python interpreter not found: $Python"
}
if (-not $OutDir) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutDir = Join-Path $Repo "efficiency_results\uda_tuned_${Target}_seed${Seed}_$stamp"
}

# Method-specific recalibration for the shared JMDA multimodal front end.
# These values come from the first unified-protocol backlight/seed42 curves:
# - DANN: keep the successful lower LR, but restore moderate domain pressure
#   and a longer budget so adaptation, rather than source-only fitting, grows.
# - DeepCORAL: place the classifier between the two previous learning speeds,
#   use two source-only epochs, then restore the gentler 0->0.3 CORAL ramp.
# - CDAN: retain the 300-update source warmup and raise only transfer weight;
#   discriminator LR stays fixed so this round isolates the changed pressure.
# - MDD: its last epoch was still improving, so extend the same 60-update
#   schedule without changing its already useful loss balance.
# - MCC: preserve 40 updates/epoch, restore 200 absolute warmup updates, reduce
#   confidence-suppressing smoothing and make the MCC term non-negligible.
$env:JMDA_DETERMINISTIC = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$arguments = @{
    Seeds = @($Seed)
    Targets = @($Target)
    IncludeMethods = $IncludeMethods
    IncludePausedMethods = $true
    Python = $Python
    OutDir = $OutDir
    ImageBatchSize = $BatchSize
    Epochs = 0
    DANNLearningRate = 0.0003
    DANNEpochs = 36
    DANNDomainLossWeight = 0.35
    DANNGrlMaxIterations = 5000
    DANNTargetEvalInterval = 1
    DANNLearningRateScheduler = 'constant'
    DANNMinimumLearningRateRatio = 0.1
    DeepCoralLearningRate = 0.0004
    DeepCoralEpochs = 30
    DeepCoralMaxWeight = 0.3
    DeepCoralWarmupEpochs = 2
    DeepCoralWeightDecay = 0.0005
    DeepCoralLabelSmoothing = 0.05
    DeepCoralTargetEvalInterval = 1
    DeepCoralHeadLearningRateMultiplier = 5.0
    DeepCoralVisualLearningRateMultiplier = 0.1
    DeepCoralInfraredLearningRateMultiplier = 1.0
    DeepCoralLearningRateScheduler = 'constant'
    DeepCoralMinimumLearningRateRatio = 0.1
    CDANLearningRate = 0.0005
    CdanIterations = 2000
    CDANHeadLearningRateMultiplier = 3.0
    CDANAdLearningRateMultiplier = 3.0
    CDANTradeOff = 0.6
    CDANSourceWarmupIterations = 300
    CDANLabelSmoothing = 0.05
    CDANCoeffMaxIterations = 6000
    CDANCoeffAlpha = 10.0
    CDANTestInterval = 100
    MDDLearningRate = 0.0025
    MDDEpochs = 40
    MDDIterationsPerEpoch = 60
    MDDTradeOff = 0.35
    MDDTradeOffWarmupEpochs = 16
    MCCLearningRate = 0.001
    MCCEpochs = 24
    MCCIterationsPerEpoch = 40
    MCCTradeOff = 0.05
    MCCTradeOffWarmupEpochs = 10
    MCCSourceWarmupEpochs = 5
    MCCLabelSmoothing = 0.10
    SkipBenchmark = $true
}
if ($DataDir) { $arguments.DataDir = $DataDir }

Write-Host 'Target-specific tuned UDA run (exact-paired VIS+IR).'
Write-Host "Target=$Target Seed=$Seed BatchSize=$BatchSize"
Write-Host "Methods=$($IncludeMethods -join ', ')"
Write-Host "Output=$OutDir"
& (Join-Path $PSScriptRoot 'run_uda_multimodal.ps1') @arguments
exit $LASTEXITCODE
