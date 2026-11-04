param(
    [ValidateSet('formal', 'smoke')]
    [string]$Mode = 'formal'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$DataRoot = if ($env:VDS2RAW_DATA_ROOT) {
    $env:VDS2RAW_DATA_ROOT
} else {
    'D:\Downloads\VDS2Raw\threeclass_ready'
}
$OutputRoot = if ($env:VDS2RAW_OUTPUT_ROOT) {
    $env:VDS2RAW_OUTPUT_ROOT
} else {
    Join-Path $RepoRoot 'runs\vds2raw'
}
$Seeds = if ($env:VDS2RAW_SEEDS) {
    @($env:VDS2RAW_SEEDS -split '\s+' | Where-Object { $_ })
} else {
    @('42')
}
$MethodOrder = if ($env:VDS2RAW_METHOD_ORDER) {
    @($env:VDS2RAW_METHOD_ORDER -split '\s+' | Where-Object { $_ })
} else {
    @('ccsa', 'dsne', 'mcft3')
}
$expectedMethods = @('ccsa', 'dsne', 'mcft3')
if ($MethodOrder.Count -lt 1 -or $MethodOrder.Count -gt 3) {
    throw 'VDS2RAW_METHOD_ORDER must contain one to three methods from: ccsa dsne mcft3.'
}
$unknownMethods = @($MethodOrder | Where-Object { $_ -notin $expectedMethods })
$duplicateMethods = @(
    $MethodOrder | Group-Object | Where-Object { $_.Count -gt 1 }
)
if ($unknownMethods.Count -or $duplicateMethods.Count) {
    throw 'VDS2RAW_METHOD_ORDER contains an unknown or duplicate method.'
}

if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "VDS2Raw data root does not exist: $DataRoot"
}

function Get-EnvInt([string]$Name, [int]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    $parsed = 0
    if (-not [int]::TryParse($value, [ref]$parsed) -or $parsed -le 0) {
        throw "$Name must be a positive integer: $value"
    }
    return $parsed
}

function Get-EnvNonNegativeInt([string]$Name, [int]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    $parsed = 0
    if (-not [int]::TryParse($value, [ref]$parsed) -or $parsed -lt 0) {
        throw "$Name must be a non-negative integer: $value"
    }
    return $parsed
}

function Get-EnvPositiveDouble([string]$Name, [double]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    $parsed = 0.0
    if (-not [double]::TryParse(
        $value,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$parsed
    ) -or $parsed -le 0) {
        throw "$Name must be a positive number: $value"
    }
    return $parsed
}

function Get-EnvBool([string]$Name, [bool]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    switch ($value.Trim().ToLowerInvariant()) {
        { $_ -in @('1', 'true', 'yes', 'on') } { return $true }
        { $_ -in @('0', 'false', 'no', 'off') } { return $false }
        default { throw "$Name must be a boolean (0/1 or true/false): $value" }
    }
}

$CcsaEpochs = Get-EnvInt 'VDS2RAW_CCSA_EPOCHS' 80
$DsneEpochs = Get-EnvInt 'VDS2RAW_DSNE_EPOCHS' 20
$McftEpochs = Get-EnvInt 'VDS2RAW_MCFT_EPOCHS' 200
$CcsaSourceEpochs = Get-EnvInt 'VDS2RAW_CCSA_SOURCE_EPOCHS' 50
$DsneSourceEpochs = Get-EnvInt 'VDS2RAW_DSNE_SOURCE_EPOCHS' 100
$CcsaBatch = Get-EnvInt 'VDS2RAW_CCSA_BATCH_SIZE' 256
$CcsaSourceBatch = Get-EnvInt 'VDS2RAW_CCSA_SOURCE_BATCH_SIZE' 32
$DsneBatch = Get-EnvInt 'VDS2RAW_DSNE_BATCH_SIZE' 32
$DsneSourceBatch = Get-EnvInt 'VDS2RAW_DSNE_SOURCE_BATCH_SIZE' 16
$McftBatch = Get-EnvInt 'VDS2RAW_MCFT_BATCH_SIZE' 32
$CcsaLr = Get-EnvPositiveDouble 'VDS2RAW_CCSA_LR' 0.005
$CcsaBackboneLrMultiplier = Get-EnvPositiveDouble 'VDS2RAW_CCSA_BACKBONE_LR_MULTIPLIER' 0.02
$CcsaSourceLr = Get-EnvPositiveDouble 'VDS2RAW_CCSA_SOURCE_LR' 0.5
$CcsaLrMinRatio = Get-EnvPositiveDouble 'VDS2RAW_CCSA_LR_MIN_RATIO' 0.01
$CcsaSourceLrMinRatio = Get-EnvPositiveDouble 'VDS2RAW_CCSA_SOURCE_LR_MIN_RATIO' 0.1
$DsneLayers = Get-EnvInt 'VDS2RAW_DSNE_LAYERS' 101
$DsneSourceWarmup = Get-EnvNonNegativeInt 'VDS2RAW_DSNE_SOURCE_HEAD_WARMUP_EPOCHS' 10
$DsneLr = Get-EnvPositiveDouble 'VDS2RAW_DSNE_LR' 0.000125
$DsneAdaptBackboneLrMultiplier = Get-EnvPositiveDouble 'VDS2RAW_DSNE_ADAPTATION_BACKBONE_LR_MULTIPLIER' 0.05
$DsneLrMinRatio = Get-EnvPositiveDouble 'VDS2RAW_DSNE_LR_MIN_RATIO' 0.1
$DsneSourceHeadLr = Get-EnvPositiveDouble 'VDS2RAW_DSNE_SOURCE_HEAD_LR' 0.005
$DsneSourceBackboneLr = Get-EnvPositiveDouble 'VDS2RAW_DSNE_SOURCE_BACKBONE_LR' 0.0005
$DsneSourceLrMinRatio = Get-EnvPositiveDouble 'VDS2RAW_DSNE_SOURCE_LR_MIN_RATIO' 0.1
$DsneWeightDecay = Get-EnvPositiveDouble 'VDS2RAW_DSNE_WEIGHT_DECAY' 0.0001
$DsneAlpha = Get-EnvPositiveDouble 'VDS2RAW_DSNE_ALPHA' 0.2
$DsneLabelSmoothing = Get-EnvPositiveDouble 'VDS2RAW_DSNE_LABEL_SMOOTHING' 0.05
$McftLr = Get-EnvPositiveDouble 'VDS2RAW_MCFT_LR' 0.0005
$McftWeightDecay = Get-EnvPositiveDouble 'VDS2RAW_MCFT_WEIGHT_DECAY' 0.0001
$McftStepSize = Get-EnvInt 'VDS2RAW_MCFT_STEP_SIZE' 20
$McftGamma = Get-EnvPositiveDouble 'VDS2RAW_MCFT_GAMMA' 0.9
$McftLabelSmoothing = Get-EnvPositiveDouble 'VDS2RAW_MCFT_LABEL_SMOOTHING' 0.1
$McftItmWeight = Get-EnvPositiveDouble 'VDS2RAW_MCFT_ITM_WEIGHT' 1.0
$McftAisDropout = Get-EnvPositiveDouble 'VDS2RAW_MCFT_AIS_DROPOUT_PROBABILITY' 0.5
$McftClassBalanced = Get-EnvBool 'VDS2RAW_MCFT_CLASS_BALANCED_TRAIN' $false
$CcsaEarlyPatience = Get-EnvNonNegativeInt 'VDS2RAW_CCSA_EARLY_STOPPING_PATIENCE' 20
$CcsaEarlyMinEpochs = Get-EnvNonNegativeInt 'VDS2RAW_CCSA_EARLY_STOPPING_MIN_EPOCHS' 30
$DsneEarlyPatience = Get-EnvNonNegativeInt 'VDS2RAW_DSNE_EARLY_STOPPING_PATIENCE' 10
$DsneEarlyMinEpochs = Get-EnvNonNegativeInt 'VDS2RAW_DSNE_EARLY_STOPPING_MIN_EPOCHS' 20
$McftEarlyPatience = Get-EnvNonNegativeInt 'VDS2RAW_MCFT_EARLY_STOPPING_PATIENCE' 25
$McftEarlyMinEpochs = Get-EnvNonNegativeInt 'VDS2RAW_MCFT_EARLY_STOPPING_MIN_EPOCHS' 60
$legacyTargetValue = [Environment]::GetEnvironmentVariable(
    'VDS2RAW_TARGET_SAMPLES_PER_CLASS'
)
$legacyTargetPerClass = if ([string]::IsNullOrWhiteSpace($legacyTargetValue)) {
    $null
} else {
    Get-EnvNonNegativeInt 'VDS2RAW_TARGET_SAMPLES_PER_CLASS' 0
}
$CcsaTargetPerClass = if ($env:VDS2RAW_CCSA_TARGET_SAMPLES_PER_CLASS) {
    Get-EnvNonNegativeInt 'VDS2RAW_CCSA_TARGET_SAMPLES_PER_CLASS' 0
} elseif ($null -ne $legacyTargetPerClass) {
    $legacyTargetPerClass
} else {
    0
}
$DsneTargetPerClass = if ($env:VDS2RAW_DSNE_TARGET_SAMPLES_PER_CLASS) {
    Get-EnvNonNegativeInt 'VDS2RAW_DSNE_TARGET_SAMPLES_PER_CLASS' 0
} elseif ($null -ne $legacyTargetPerClass) {
    $legacyTargetPerClass
} else {
    0
}
$McftEvalFrequency = Get-EnvInt 'VDS2RAW_MCFT_EVAL_FREQUENCY' 1
if ($Mode -eq 'formal' -and (
    ($MethodOrder -contains 'ccsa' -and $CcsaTargetPerClass -ne 0) -or
    ($MethodOrder -contains 'dsne' -and $DsneTargetPerClass -ne 0))) {
    throw ('Formal VDS2Raw runs require all target_train labels: set both ' +
        'target-samples-per-class values to 0.')
}
$CcsaSourceCheckpoint = $env:VDS2RAW_CCSA_SOURCE_CHECKPOINT
$DsneSourceCheckpoint = $env:VDS2RAW_DSNE_SOURCE_CHECKPOINT
$CcsaPretrainedBackbone = $env:VDS2RAW_CCSA_PRETRAINED_BACKBONE
$DsnePretrainedBackbone = $env:VDS2RAW_DSNE_PRETRAINED_BACKBONE
if ($CcsaSourceCheckpoint -and $CcsaPretrainedBackbone) {
    throw 'Set only one of VDS2RAW_CCSA_SOURCE_CHECKPOINT and VDS2RAW_CCSA_PRETRAINED_BACKBONE.'
}
if ($DsneSourceCheckpoint -and $DsnePretrainedBackbone) {
    throw 'Set only one of VDS2RAW_DSNE_SOURCE_CHECKPOINT and VDS2RAW_DSNE_PRETRAINED_BACKBONE.'
}
$MaxBatches = 0
$PretrainedArgs = @()
if ($Mode -eq 'smoke') {
    $CcsaEpochs = 1
    $DsneEpochs = 1
    $McftEpochs = 1
    $CcsaSourceEpochs = 1
    $DsneSourceEpochs = 1
    $CcsaBatch = 8
    $CcsaSourceBatch = 4
    $DsneBatch = 4
    $DsneSourceBatch = 4
    $McftBatch = 4
    $DsneLayers = 101
    $DsneSourceWarmup = 0
    $McftEvalFrequency = 1
    $CcsaEarlyPatience = 0
    $DsneEarlyPatience = 0
    $McftEarlyPatience = 0
    $MaxBatches = 1
    $PretrainedArgs = @('--no-pretrained')
}

$RunId = 'vds2raw_{0}_{1}_{2}' -f $Mode, (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID
$RunDir = Join-Path $OutputRoot $RunId
$ArtifactDir = Join-Path $RunDir 'artifacts'
$LogDir = Join-Path $RunDir 'logs'
$SummaryDir = Join-Path $RunDir 'summary'
$StatsFile = Join-Path $RunDir 'normalization_stats.json'
$AuditFile = Join-Path $RunDir 'dataset_audit.json'
New-Item -ItemType Directory -Force -Path $ArtifactDir, $LogDir, $SummaryDir | Out-Null

$config = [ordered]@{
    mode = $Mode
    data_root = (Resolve-Path -LiteralPath $DataRoot).Path
    seeds = $Seeds
    method_order = $MethodOrder
    protocol = 'all source_train and target_train labels for every method; target_test peak checkpoint selection (biased test-selected estimate)'
    ccsa_epochs = $CcsaEpochs
    ccsa_source_epochs = $CcsaSourceEpochs
    ccsa_batch_size = $CcsaBatch
    ccsa_source_batch_size = $CcsaSourceBatch
    ccsa_lr = $CcsaLr
    ccsa_backbone_lr_multiplier = $CcsaBackboneLrMultiplier
    ccsa_source_lr = $CcsaSourceLr
    ccsa_lr_min_ratio = $CcsaLrMinRatio
    ccsa_source_lr_min_ratio = $CcsaSourceLrMinRatio
    dsne_epochs = $DsneEpochs
    dsne_source_epochs = $DsneSourceEpochs
    dsne_batch_size = $DsneBatch
    dsne_source_batch_size = $DsneSourceBatch
    dsne_lr = $DsneLr
    dsne_adaptation_backbone_lr_multiplier = $DsneAdaptBackboneLrMultiplier
    dsne_lr_min_ratio = $DsneLrMinRatio
    dsne_source_head_warmup_epochs = $DsneSourceWarmup
    dsne_source_head_lr = $DsneSourceHeadLr
    dsne_source_backbone_lr = $DsneSourceBackboneLr
    dsne_source_lr_min_ratio = $DsneSourceLrMinRatio
    dsne_weight_decay = $DsneWeightDecay
    dsne_alpha = $DsneAlpha
    dsne_label_smoothing = $DsneLabelSmoothing
    mcft_epochs = $McftEpochs
    mcft_batch_size = $McftBatch
    mcft_lr = $McftLr
    mcft_weight_decay = $McftWeightDecay
    mcft_step_size = $McftStepSize
    mcft_gamma = $McftGamma
    mcft_label_smoothing = $McftLabelSmoothing
    mcft_itm_weight = $McftItmWeight
    mcft_ais_dropout_probability = $McftAisDropout
    mcft_class_balanced_train = $McftClassBalanced
    dsne_layers = $DsneLayers
    ccsa_pair_protocol = 'fixed exhaustive official pair table'
    ccsa_target_samples_per_class = $CcsaTargetPerClass
    dsne_target_samples_per_class = $DsneTargetPerClass
    dsne_feature_normalization = $true
    ccsa_early_stopping = @{
        patience = $CcsaEarlyPatience; min_epochs = $CcsaEarlyMinEpochs
    }
    dsne_early_stopping = @{
        patience = $DsneEarlyPatience; min_epochs = $DsneEarlyMinEpochs
    }
    mcft_early_stopping = @{
        patience = $McftEarlyPatience; min_epochs = $McftEarlyMinEpochs
    }
    model_selection = 'best target_test ACC with macro-F1 tie-break'
    test_selection_bias = $true
    ccsa_source_checkpoint = if ($CcsaSourceCheckpoint) { $CcsaSourceCheckpoint } else { $null }
    dsne_source_checkpoint = if ($DsneSourceCheckpoint) { $DsneSourceCheckpoint } else { $null }
    ccsa_pretrained_backbone = if ($CcsaPretrainedBackbone) { $CcsaPretrainedBackbone } else { $null }
    dsne_pretrained_backbone = if ($DsnePretrainedBackbone) { $DsnePretrainedBackbone } else { $null }
}
$config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RunDir 'run_config.json') -Encoding utf8

Write-Host "Run directory: $RunDir"
Write-Host 'Auditing all 271 NPZ files and fitting train-only image statistics...'
& $PythonBin (Join-Path $RepoRoot 'vds2raw\audit.py') `
    --data-root $DataRoot --stats-file $StatsFile --report-file $AuditFile
if ($LASTEXITCODE -ne 0) { throw 'VDS2Raw audit failed; training was not started.' }

function Invoke-Method([string]$Tag, [string[]]$Arguments) {
    $LogPath = Join-Path $LogDir "$Tag.log"
    Write-Host ">>> $Tag"
    & $PythonBin @Arguments *> $LogPath
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Get-Content -LiteralPath $LogPath -Tail 40
        throw "$Tag failed with exit code $exitCode. Log: $LogPath"
    }
    Write-Host "    completed; log: $LogPath"
}

foreach ($seed in $Seeds) {
    $common = @(
        '--data-root', $DataRoot,
        '--save-dir', $ArtifactDir,
        '--stats-file', $StatsFile,
        '--seed', $seed,
        '--num-workers', '0',
        '--skip-audit'
    )
    $smokeArgs = if ($MaxBatches) { @('--max-train-batches', "$MaxBatches") } else { @() }
    $ccsaCheckpointArgs = if ($CcsaSourceCheckpoint) {
        @('--source-checkpoint', $CcsaSourceCheckpoint)
    } else { @() }
    $dsneCheckpointArgs = if ($DsneSourceCheckpoint) {
        @('--source-checkpoint', $DsneSourceCheckpoint)
    } else { @() }
    $ccsaBackboneArgs = if ($CcsaPretrainedBackbone) {
        @('--pretrained-backbone', $CcsaPretrainedBackbone)
    } else { @() }
    $dsneBackboneArgs = if ($DsnePretrainedBackbone) {
        @('--pretrained-backbone', $DsnePretrainedBackbone)
    } else { @() }
    $ccsaArguments = @(
        (Join-Path $RepoRoot 'vds2raw\train_ccsa.py'),
        '--epochs', "$CcsaEpochs", '--source-epochs', "$CcsaSourceEpochs",
        '--batch-size', "$CcsaBatch", '--source-batch-size', "$CcsaSourceBatch",
        '--lr', "$CcsaLr", '--source-lr', "$CcsaSourceLr",
        '--backbone-lr-multiplier', "$CcsaBackboneLrMultiplier",
        '--lr-min-ratio', "$CcsaLrMinRatio",
        '--source-lr-min-ratio', "$CcsaSourceLrMinRatio", '--class-balanced-ce',
        '--early-stopping-patience', "$CcsaEarlyPatience",
        '--early-stopping-min-epochs', "$CcsaEarlyMinEpochs",
        '--target-samples-per-class', "$CcsaTargetPerClass"
    ) + $common + $smokeArgs + $PretrainedArgs + $ccsaCheckpointArgs + $ccsaBackboneArgs
    $dsneArguments = @(
        (Join-Path $RepoRoot 'vds2raw\train_dsne.py'),
        '--epochs', "$DsneEpochs", '--batch-size', "$DsneBatch",
        '--source-epochs', "$DsneSourceEpochs",
        '--source-batch-size', "$DsneSourceBatch",
        '--lr', "$DsneLr", '--lr-min-ratio', "$DsneLrMinRatio",
        '--adaptation-backbone-lr-multiplier', "$DsneAdaptBackboneLrMultiplier",
        '--source-head-warmup-epochs', "$DsneSourceWarmup",
        '--source-head-lr', "$DsneSourceHeadLr",
        '--source-backbone-lr', "$DsneSourceBackboneLr",
        '--source-lr-min-ratio', "$DsneSourceLrMinRatio",
        '--weight-decay', "$DsneWeightDecay", '--alpha', "$DsneAlpha",
        '--label-smoothing', "$DsneLabelSmoothing", '--dropout',
        '--early-stopping-patience', "$DsneEarlyPatience",
        '--early-stopping-min-epochs', "$DsneEarlyMinEpochs",
        '--augmentation-strength', 'light', '--class-balanced-source',
        '--class-balanced-batches', '--nlayers', "$DsneLayers",
        '--target-samples-per-class', "$DsneTargetPerClass", '--fn'
    ) + $common + $smokeArgs + $PretrainedArgs + $dsneCheckpointArgs + $dsneBackboneArgs
    $mcftArguments = @(
        (Join-Path $RepoRoot 'vds2raw\train_mcft.py'),
        '--epochs', "$McftEpochs", '--batch-size', "$McftBatch",
        '--lr', "$McftLr", '--weight-decay', "$McftWeightDecay",
        '--step-size', "$McftStepSize", '--gamma', "$McftGamma",
        '--label-smoothing', "$McftLabelSmoothing",
        '--loss-itm-weight', "$McftItmWeight",
        '--ais-dropout-probability', "$McftAisDropout",
        $(if ($McftClassBalanced) { '--class-balanced-train' } else { '--no-class-balanced-train' }),
        '--eval-frequency', "$McftEvalFrequency",
        '--early-stopping-patience', "$McftEarlyPatience",
        '--early-stopping-min-epochs', "$McftEarlyMinEpochs"
    ) + $common + $smokeArgs
    foreach ($method in $MethodOrder) {
        switch ($method) {
            'ccsa' { Invoke-Method -Tag "ccsa_seed$seed" -Arguments $ccsaArguments }
            'dsne' { Invoke-Method -Tag "dsne_seed$seed" -Arguments $dsneArguments }
            'mcft3' { Invoke-Method -Tag "mcft3_seed$seed" -Arguments $mcftArguments }
        }
    }
}

if ($env:VDS2RAW_RUN_COMPLEXITY_BENCHMARK -ne '0') {
    $benchmarkMethodMap = @{
        ccsa = 'vds_ccsa'
        dsne = 'vds_dsne'
        mcft3 = 'vds_mcft3'
    }
    $benchmarkMethods = ($MethodOrder | ForEach-Object {
        $benchmarkMethodMap[$_]
    }) -join ','
    $benchmarkArguments = @(
        (Join-Path $RepoRoot 'tools\benchmark_efficiency.py'),
        '--methods', $benchmarkMethods,
        '--num-classes', '3', '--protocol', 'benchmark',
        '--benchmark_train_batch_size', '1',
        '--efficiency_output_dir', (Join-Path $RunDir 'efficiency')
    )
    if ($Mode -eq 'smoke') { $benchmarkArguments += '--smoke' }
    Invoke-Method -Tag 'complexity' -Arguments $benchmarkArguments
}

& $PythonBin (Join-Path $RepoRoot 'vds2raw\summarize.py') `
    --results-dir $ArtifactDir --output-dir $SummaryDir
if ($LASTEXITCODE -ne 0) { throw 'Result summary failed.' }
Write-Host 'All VDS2Raw methods completed.'
Write-Host "Per-run and summary metrics: $SummaryDir"
Write-Host "Checkpoints and detailed records: $ArtifactDir"

