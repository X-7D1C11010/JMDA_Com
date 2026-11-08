# Official single-modal UDA suite for the JMDA weather dataset.
# VIS keeps each method's published image pipeline. IR changes only the input
# list, grayscale ResNet-50 stem, and one-channel normalization.
param(
    [ValidateSet('smoke', 'formal')]
    [string]$Mode = 'formal',
    [ValidateSet('vis', 'ir')]
    [string[]]$Modalities = @('vis', 'ir'),
    [string[]]$Targets = @('backlight', 'rainy', 'foggy', 'dark'),
    [string[]]$Methods = @('DANN', 'DeepCORAL', 'CDAN', 'MDD', 'MCC'),
    [int[]]$Seeds = @(42),
    [string]$Python = 'D:\Anaconda\envs\pytorch\python.exe',
    [string]$DataRoot = 'D:\Code\JMDA-Net\Data',
    [string]$OutDir = '',
    [int]$NumWorkers = 2,
    [int]$EvalNumWorkers = 0,
    [int]$PrefetchFactor = 4,
    [int]$WorkerTimeoutSeconds = 300
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutDir = Join-Path $Repo "efficiency_results\official_single_modal_$stamp"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python interpreter not found: $Python"
}
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "Dataset root not found: $DataRoot"
}

Write-Host 'Checking official ResNet-50 and AlexNet weights ...'
& $Python -B -u (Join-Path $Repo 'tools\download_official_uda_weights.py') `
    --output-dir (Join-Path $Repo 'pretrained_weights') `
    --only resnet50 alexnet
if ($LASTEXITCODE -ne 0) {
    throw "Failed to prepare official pretrained weights (exit=$LASTEXITCODE)"
}

$targetArgs = @($Targets | ForEach-Object { $_.Trim().ToLowerInvariant() })
$methodArgs = @($Methods | ForEach-Object { $_.Trim() })

foreach ($modalityValue in $Modalities) {
    $modality = $modalityValue.ToLowerInvariant()
    $listDir = Join-Path $OutDir "lists_$modality"
    New-Item -ItemType Directory -Force -Path $listDir | Out-Null

    Write-Host "Preparing complete $($modality.ToUpperInvariant()) cohorts ..."
    & $Python -B -u (Join-Path $Repo 'tools\prepare_paired_protocol_lists.py') `
        --data-root $DataRoot `
        --output-dir $listDir `
        --targets @targetArgs `
        --modality $modality
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare $modality data lists (exit=$LASTEXITCODE)"
    }

    $runDir = Join-Path $OutDir $modality
    $arguments = @{
        Python = $Python
        OutDir = $runDir
        DataDir = $listDir
        Seeds = $Seeds
        Targets = $targetArgs
        IncludeMethods = $methodArgs
        IncludePausedMethods = $true
        SingleModality = $modality
        OfficialSingleModal = $true
        NumWorkers = $NumWorkers
        EvalNumWorkers = $EvalNumWorkers
        PrefetchFactor = $PrefetchFactor
        WorkerTimeoutSeconds = $WorkerTimeoutSeconds
        SkipBenchmark = $true
        DANNEpochs = 20
        DANNIterationsPerEpoch = 1000
        DANNLearningRate = 0.01
        DANNTargetEvalInterval = 1
        DeepCoralTargetEvalInterval = 1
        CdanIterations = 100004
        CDANTestInterval = 500
        MDDIterationsPerEpoch = 1000
        MCCIterationsPerEpoch = 1000
        MCCSourceWarmupEpochs = 0
    }
    if ($Mode -eq 'smoke') {
        $arguments.Epochs = 1
        $arguments.DANNIterationsPerEpoch = 1
        $arguments.CdanIterations = 2
        $arguments.CDANTestInterval = 1
        $arguments.MDDIterationsPerEpoch = 1
        $arguments.MCCIterationsPerEpoch = 1
    }
    Write-Host "Running official $($modality.ToUpperInvariant()) suite ..."
    & (Join-Path $PSScriptRoot 'run_uda_sunny.ps1') @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$modality suite failed (exit=$LASTEXITCODE)"
    }
}

Write-Host "Official single-modal suite finished: $OutDir"
