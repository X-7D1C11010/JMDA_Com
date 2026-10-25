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

$CcsaEpochs = Get-EnvInt 'VDS2RAW_CCSA_EPOCHS' 80
$DsneEpochs = Get-EnvInt 'VDS2RAW_DSNE_EPOCHS' 20
$McftEpochs = Get-EnvInt 'VDS2RAW_MCFT_EPOCHS' 500
$CcsaSourceEpochs = Get-EnvInt 'VDS2RAW_CCSA_SOURCE_EPOCHS' 50
$DsneSourceEpochs = Get-EnvInt 'VDS2RAW_DSNE_SOURCE_EPOCHS' 50
$CcsaBatch = Get-EnvInt 'VDS2RAW_CCSA_BATCH_SIZE' 256
$DsneBatch = Get-EnvInt 'VDS2RAW_DSNE_BATCH_SIZE' 32
$McftBatch = Get-EnvInt 'VDS2RAW_MCFT_BATCH_SIZE' 32
$DsneLayers = Get-EnvInt 'VDS2RAW_DSNE_LAYERS' 101
$CcsaPairs = Get-EnvInt 'VDS2RAW_CCSA_PAIRS_PER_EPOCH' 768
$TargetPerClass = Get-EnvInt 'VDS2RAW_TARGET_SAMPLES_PER_CLASS' 10
$McftEvalFrequency = Get-EnvInt 'VDS2RAW_MCFT_EVAL_FREQUENCY' 10
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
    $DsneBatch = 4
    $McftBatch = 4
    $DsneLayers = 101
    $CcsaPairs = 24
    $McftEvalFrequency = 1
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
    protocol = 'source_train + fixed-shot target_train; target_val selection; target_test once per predeclared method/budget'
    ccsa_epochs = $CcsaEpochs
    ccsa_source_epochs = $CcsaSourceEpochs
    dsne_epochs = $DsneEpochs
    dsne_source_epochs = $DsneSourceEpochs
    mcft_epochs = $McftEpochs
    ccsa_batch_size = $CcsaBatch
    dsne_batch_size = $DsneBatch
    mcft_batch_size = $McftBatch
    dsne_layers = $DsneLayers
    ccsa_pairs_per_epoch = $CcsaPairs
    target_samples_per_class = $TargetPerClass
    dsne_feature_normalization = $true
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
        '--batch-size', "$CcsaBatch", '--pairs-per-epoch', "$CcsaPairs",
        '--target-samples-per-class', "$TargetPerClass"
    ) + $common + $smokeArgs + $PretrainedArgs + $ccsaCheckpointArgs + $ccsaBackboneArgs
    Invoke-Method -Tag "ccsa_seed$seed" -Arguments $ccsaArguments
    $dsneArguments = @(
        (Join-Path $RepoRoot 'vds2raw\train_dsne.py'),
        '--epochs', "$DsneEpochs", '--batch-size', "$DsneBatch",
        '--source-epochs', "$DsneSourceEpochs", '--nlayers', "$DsneLayers",
        '--target-samples-per-class', "$TargetPerClass", '--fn'
    ) + $common + $smokeArgs + $PretrainedArgs + $dsneCheckpointArgs + $dsneBackboneArgs
    Invoke-Method -Tag "dsne_seed$seed" -Arguments $dsneArguments
    $mcftArguments = @(
        (Join-Path $RepoRoot 'vds2raw\train_mcft.py'),
        '--epochs', "$McftEpochs", '--batch-size', "$McftBatch",
        '--eval-frequency', "$McftEvalFrequency"
    ) + $common + $smokeArgs
    Invoke-Method -Tag "mcft3_seed$seed" -Arguments $mcftArguments
}

& $PythonBin (Join-Path $RepoRoot 'vds2raw\summarize.py') `
    --results-dir $ArtifactDir --output-dir $SummaryDir
if ($LASTEXITCODE -ne 0) { throw 'Result summary failed.' }
Write-Host 'All VDS2Raw methods completed.'
Write-Host "Per-run and summary metrics: $SummaryDir"
Write-Host "Checkpoints and detailed records: $ArtifactDir"

