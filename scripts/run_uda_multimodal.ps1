param(
    [int[]]$Seeds = @(42),
    [string[]]$Targets = @('backlight', 'rainy', 'foggy', 'dark'),
    [string[]]$IncludeMethods = @(),
    [string]$Python = 'D:\Anaconda\envs\pytorch\python.exe',
    [string]$OutDir = '',
    [string]$DataDir = '',
    [int]$CdanIterations = 2000,
    [int]$ImageBatchSize = 16,
    [int]$Epochs = 0,
    [double]$DANNLearningRate = 0.001,
    [double]$DeepCoralLearningRate = 0.001,
    [double]$CDANLearningRate = 0.001,
    [double]$MDDLearningRate = 0.004,
    [double]$MCCLearningRate = 0.005,
    [double]$PDALearningRate = 0.003,
    [int]$DANNEpochs = 100,
    [int]$DeepCoralEpochs = 20,
    [int]$MDDEpochs = 20,
    [int]$MCCEpochs = 20,
    [int]$PDAEpochs = 20,
    [double]$DANNDomainLossWeight = 1.0,
    [int]$DANNTargetEvalInterval = 0,
    [double]$DeepCoralMaxWeight = 1.0,
    [double]$DeepCoralWeightDecay = 0.0,
    [int]$DeepCoralTargetEvalInterval = 0,
    [double]$CDANHeadLearningRateMultiplier = 10.0,
    [int]$CDANCoeffMaxIterations = 0,
    [int]$CDANTestInterval = 500,
    [int]$MDDIterationsPerEpoch = 1000,
    [double]$MDDTradeOff = 1.0,
    [int]$MCCIterationsPerEpoch = 1000,
    [double]$MCCTradeOff = 1.0,
    [int]$MCCTradeOffWarmupEpochs = 0,
    [int]$PDAWarmupEpochs = 1,
    [double]$PDAConfidenceThreshold = 0.9,
    [int]$PDATargetEvalInterval = 0,
    [int]$NumWorkers = 2,
    [int]$EvalNumWorkers = 0,
    [int]$PrefetchFactor = 4,
    [int]$WorkerTimeoutSeconds = 300,
    [switch]$IncludePausedMethods,
    [switch]$SkipBenchmark
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutDir = Join-Path $Repo "efficiency_results\uda_multimodal_$stamp"
}
if (-not $DataDir) {
    $dataRoot = $env:JMDA_DATA_ROOT
    if (-not $dataRoot) { $dataRoot = 'D:\Code\JMDA-Net\Data' }
    if (-not (Test-Path -LiteralPath $dataRoot -PathType Container)) {
        throw "JMDA data root not found: $dataRoot. Set JMDA_DATA_ROOT first."
    }
    $DataDir = Join-Path $OutDir 'prepared_exact_pairs'
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
    $prepareArgs = @('-B', '-u',
        (Join-Path $Repo 'tools\prepare_paired_protocol_lists.py'),
        '--data-root', $dataRoot,
        '--output-dir', $DataDir,
        '--targets') + $Targets + @('--modality', 'paired')
    & $Python @prepareArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare exact-paired protocol lists (exit=$LASTEXITCODE)."
    }
}

$needsResNet50 = ($IncludeMethods.Count -eq 0) -or
    (@($IncludeMethods | Where-Object { $_ -ine 'PDA' }).Count -gt 0)
$resNet50Weight = Join-Path $Repo 'pretrained_weights\torch\hub\checkpoints\resnet50-0676ba61.pth'
if ($needsResNet50 -and -not (Test-Path -LiteralPath $resNet50Weight -PathType Leaf)) {
    throw ("Missing ResNet-50 ImageNet weight: {0}. Run: & '{1}' tools\download_official_uda_weights.py" -f `
           $resNet50Weight, $Python)
}

$arguments = @{
    Seeds = $Seeds
    Targets = $Targets
    Python = $Python
    OutDir = $OutDir
    DataDir = $DataDir
    CdanIterations = $CdanIterations
    Epochs = $Epochs
    DANNLearningRate = $DANNLearningRate
    DeepCoralLearningRate = $DeepCoralLearningRate
    CDANLearningRate = $CDANLearningRate
    MDDLearningRate = $MDDLearningRate
    MCCLearningRate = $MCCLearningRate
    PDALearningRate = $PDALearningRate
    DANNEpochs = $DANNEpochs
    DeepCoralEpochs = $DeepCoralEpochs
    MDDEpochs = $MDDEpochs
    MCCEpochs = $MCCEpochs
    PDAEpochs = $PDAEpochs
    DANNDomainLossWeight = $DANNDomainLossWeight
    DANNTargetEvalInterval = $DANNTargetEvalInterval
    DeepCoralMaxWeight = $DeepCoralMaxWeight
    DeepCoralWeightDecay = $DeepCoralWeightDecay
    DeepCoralTargetEvalInterval = $DeepCoralTargetEvalInterval
    CDANHeadLearningRateMultiplier = $CDANHeadLearningRateMultiplier
    CDANCoeffMaxIterations = $CDANCoeffMaxIterations
    CDANTestInterval = $CDANTestInterval
    MDDIterationsPerEpoch = $MDDIterationsPerEpoch
    MDDTradeOff = $MDDTradeOff
    MCCIterationsPerEpoch = $MCCIterationsPerEpoch
    MCCTradeOff = $MCCTradeOff
    MCCTradeOffWarmupEpochs = $MCCTradeOffWarmupEpochs
    PDAWarmupEpochs = $PDAWarmupEpochs
    PDAConfidenceThreshold = $PDAConfidenceThreshold
    PDATargetEvalInterval = $PDATargetEvalInterval
    NumWorkers = $NumWorkers
    EvalNumWorkers = $EvalNumWorkers
    PrefetchFactor = $PrefetchFactor
    WorkerTimeoutSeconds = $WorkerTimeoutSeconds
    ImageBackbone = 'ResNet50'
    ImageBatchSize = $ImageBatchSize
    EnhancedAugmentation = $true
    Multimodal = $true
    EarlyStoppingPatience = 0
    SkipBenchmark = [bool]$SkipBenchmark
}
if ($IncludeMethods.Count -gt 0) { $arguments.IncludeMethods = $IncludeMethods }
if ($IncludePausedMethods) { $arguments.IncludePausedMethods = $true }

Write-Host 'Protocol: exact-paired VIS+IR; final model in memory only; no target-val selection; no checkpoint saving.'
Write-Host "Per-domain batch size: $ImageBatchSize"
Write-Host "CDAN iterations: $CdanIterations"
& (Join-Path $PSScriptRoot 'run_uda_sunny.ps1') @arguments
exit $LASTEXITCODE
