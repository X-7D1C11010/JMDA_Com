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
# - DANN: alignment became unstable when GRL/domain pressure rose; use weaker,
#   slower adversarial pressure and keep the LR constant for the random IR arm.
# - DeepCORAL: a 10x head fitted source by epoch 2; slow the head and increase
#   the gradual CORAL ceiling so classification and alignment learn together.
# - CDAN: the random IR arm needs source-only stabilization before conditional
#   adversarial training; the discriminator is also reduced from 10x to 3x.
# - MDD: retain the successful direction but reduce update density/transfer
#   dominance after the source classifier has converged.
# - MCC: target accuracy fell before MCC started, so reduce LR and updates per
#   epoch instead of extending the source-only warmup.
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
    DANNEpochs = 30
    DANNDomainLossWeight = 0.25
    DANNGrlMaxIterations = 6000
    DANNTargetEvalInterval = 1
    DANNLearningRateScheduler = 'constant'
    DANNMinimumLearningRateRatio = 0.1
    DeepCoralLearningRate = 0.0003
    DeepCoralEpochs = 24
    DeepCoralMaxWeight = 0.6
    DeepCoralWarmupEpochs = 0
    DeepCoralWeightDecay = 0.001
    DeepCoralLabelSmoothing = 0.10
    DeepCoralTargetEvalInterval = 1
    DeepCoralHeadLearningRateMultiplier = 3.0
    DeepCoralVisualLearningRateMultiplier = 0.1
    DeepCoralInfraredLearningRateMultiplier = 1.0
    DeepCoralLearningRateScheduler = 'constant'
    DeepCoralMinimumLearningRateRatio = 0.1
    CDANLearningRate = 0.0005
    CdanIterations = 2000
    CDANHeadLearningRateMultiplier = 3.0
    CDANAdLearningRateMultiplier = 3.0
    CDANTradeOff = 0.3
    CDANSourceWarmupIterations = 300
    CDANLabelSmoothing = 0.05
    CDANCoeffMaxIterations = 6000
    CDANCoeffAlpha = 10.0
    CDANTestInterval = 100
    MDDLearningRate = 0.0025
    MDDEpochs = 32
    MDDIterationsPerEpoch = 60
    MDDTradeOff = 0.35
    MDDTradeOffWarmupEpochs = 16
    MCCLearningRate = 0.001
    MCCEpochs = 20
    MCCIterationsPerEpoch = 40
    MCCTradeOff = 0.01
    MCCTradeOffWarmupEpochs = 12
    MCCSourceWarmupEpochs = 2
    MCCLabelSmoothing = 0.15
    SkipBenchmark = $true
}
if ($DataDir) { $arguments.DataDir = $DataDir }

Write-Host 'Target-specific tuned UDA run (exact-paired VIS+IR).'
Write-Host "Target=$Target Seed=$Seed BatchSize=$BatchSize"
Write-Host "Methods=$($IncludeMethods -join ', ')"
Write-Host "Output=$OutDir"
& (Join-Path $PSScriptRoot 'run_uda_multimodal.ps1') @arguments
exit $LASTEXITCODE
