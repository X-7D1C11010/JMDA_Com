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

# Settings are deliberately method-specific and are based on the observed
# backlight/seed42 curves, rather than copied from one benchmark recipe:
# - DANN: slowing the GRL ramp from 800 to 1000 updates reduced ACC from 75.00
#   to 69.44.  Reverse that change and use a slightly faster 700-update ramp;
#   retain the proven loss scale and evaluate every epoch.
# - DeepCORAL: the three source-only warmup epochs reduced ACC from 72.22 to
#   69.44.  Restore immediate CORAL alignment and add mild source label
#   smoothing to address the observed near-100% source fit and target swings.
# - CDAN: keep the capable 3x head, 10x discriminator and trade-off 1, but
#   current slower ramp improved at the final update.  Extend it modestly while
#   keeping approximately the same terminal GRL strength and 100-update evals.
# - MDD: the lower update density plus gradual transfer weighting improved ACC
#   by 5.56 points and removed the isolated early peak.  Continue with 80
#   updates/epoch and a slightly gentler/longer transfer ramp at similar total
#   update count.
# - MCC: the higher LR/transfer weight and shorter warmup regressed.  Restore
#   the preceding recipe, extend its late useful region, and add source label
#   smoothing because source CE reached near-zero by epoch two.
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
    DANNLearningRate = 0.0005
    DANNEpochs = 22
    DANNDomainLossWeight = 0.5
    DANNGrlMaxIterations = 700
    DANNTargetEvalInterval = 1
    DANNLearningRateScheduler = 'constant'
    DANNMinimumLearningRateRatio = 0.1
    DeepCoralLearningRate = 0.0005
    DeepCoralEpochs = 30
    DeepCoralMaxWeight = 0.3
    DeepCoralWarmupEpochs = 0
    DeepCoralWeightDecay = 0.0005
    DeepCoralLabelSmoothing = 0.05
    DeepCoralTargetEvalInterval = 1
    DeepCoralHeadLearningRateMultiplier = 10.0
    DeepCoralLearningRateScheduler = 'constant'
    DeepCoralMinimumLearningRateRatio = 0.05
    CDANLearningRate = 0.0005
    CdanIterations = 1400
    CDANHeadLearningRateMultiplier = 3.0
    CDANAdLearningRateMultiplier = 10.0
    CDANTradeOff = 1.0
    CDANCoeffMaxIterations = 12000
    CDANCoeffAlpha = 10.0
    CDANTestInterval = 100
    MDDLearningRate = 0.0025
    MDDEpochs = 36
    MDDIterationsPerEpoch = 80
    MDDTradeOff = 0.45
    MDDTradeOffWarmupEpochs = 12
    MCCLearningRate = 0.002
    MCCEpochs = 26
    MCCIterationsPerEpoch = 100
    MCCTradeOff = 0.15
    MCCTradeOffWarmupEpochs = 12
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
