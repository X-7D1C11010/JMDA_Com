param(
    [string]$Target = 'backlight',
    [int]$Seed = 42,
    [int]$BatchSize = 16,
    [string[]]$IncludeMethods = @('DANN', 'DeepCORAL', 'CDAN', 'MDD', 'MCC', 'PDA'),
    [string]$Python = 'D:\Anaconda\envs\pytorch\python.exe',
    [string]$OutDir = '',
    [string]$DataDir = ''
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$validTargets = @('backlight', 'rainy', 'foggy', 'dark')
$validMethods = @('DANN', 'DeepCORAL', 'CDAN', 'MDD', 'MCC', 'PDA')
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
# - DANN: retain the proven 5e-4 LR, reduce adversarial dominance.
# - DeepCORAL: slower/longer fitting and a lower final CORAL weight.
# - CDAN: slow the 10x head, extend useful training, match the GRL ramp to it.
# - MDD: retain the configuration whose peak was late (epoch 19).
# - MCC: remove 143x/epoch target recycling and warm a weaker confusion loss.
# - PDA: lower prompt LR, longer warmup, slightly relax pseudo-label confidence.
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
    DANNEpochs = 100
    DANNDomainLossWeight = 0.5
    DANNTargetEvalInterval = 5
    DeepCoralLearningRate = 0.0005
    DeepCoralEpochs = 30
    DeepCoralMaxWeight = 0.3
    DeepCoralWeightDecay = 0.0005
    DeepCoralTargetEvalInterval = 1
    CDANLearningRate = 0.0005
    CdanIterations = 3000
    CDANHeadLearningRateMultiplier = 3.0
    CDANCoeffMaxIterations = 3000
    CDANTestInterval = 250
    MDDLearningRate = 0.004
    MDDEpochs = 20
    MDDIterationsPerEpoch = 1000
    MDDTradeOff = 1.0
    MCCLearningRate = 0.0025
    MCCEpochs = 30
    MCCIterationsPerEpoch = 100
    MCCTradeOff = 0.2
    MCCTradeOffWarmupEpochs = 10
    PDALearningRate = 0.0015
    PDAEpochs = 30
    PDAWarmupEpochs = 3
    PDAConfidenceThreshold = 0.85
    PDATargetEvalInterval = 1
    SkipBenchmark = $true
}
if ($DataDir) { $arguments.DataDir = $DataDir }

Write-Host 'Target-specific tuned UDA run (exact-paired VIS+IR).'
Write-Host "Target=$Target Seed=$Seed BatchSize=$BatchSize"
Write-Host "Methods=$($IncludeMethods -join ', ')"
Write-Host "Output=$OutDir"
& (Join-Path $PSScriptRoot 'run_uda_multimodal.ps1') @arguments
exit $LASTEXITCODE
