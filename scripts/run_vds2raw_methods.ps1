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
$Seeds = if ($env:VDS2RAW_SEEDS) { $env:VDS2RAW_SEEDS -split '\s+' } else { @('42') }
$MethodOrder = if ($env:VDS2RAW_METHOD_ORDER) {
    $env:VDS2RAW_METHOD_ORDER -split '\s+'
} else {
    @('tir')
}
if ($MethodOrder.Count -lt 1 -or $MethodOrder.Count -gt 2) {
    throw 'VDS2RAW_METHOD_ORDER must contain one or two methods: mcft3 tir.'
}
foreach ($method in $MethodOrder) {
    if ($method -notin @('mcft3', 'tir')) { throw "Unknown method: $method" }
}
if (($MethodOrder | Select-Object -Unique).Count -ne $MethodOrder.Count) {
    throw 'VDS2RAW_METHOD_ORDER cannot contain duplicates.'
}

function Env-OrDefault([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ($value) { return $value }
    return $Default
}

$McftEpochs = [int](Env-OrDefault 'VDS2RAW_MCFT_EPOCHS' '200')
$McftBatch = [int](Env-OrDefault 'VDS2RAW_MCFT_BATCH_SIZE' '32')
$TirEpochs = [int](Env-OrDefault 'VDS2RAW_TIR_EPOCHS' '39')
$TirBatch = [int](Env-OrDefault 'VDS2RAW_TIR_BATCH_SIZE' '32')
$TirWarmupEpochs = [int](Env-OrDefault 'VDS2RAW_TIR_WARMUP_EPOCHS' '5')
$TirModalityWarmupEpochs = [int](Env-OrDefault 'VDS2RAW_TIR_MODALITY_WARMUP_EPOCHS' '25')
$TirTextUnfreezeEpoch = [int](Env-OrDefault 'VDS2RAW_TIR_TEXT_UNFREEZE_EPOCH' '40')
$ClipCheckpoint = Env-OrDefault 'VDS2RAW_TIR_CLIP_CHECKPOINT' `
    (Join-Path $RepoRoot 'pretrained_weights\clip\ViT-B-16.pt')
if ('tir' -in $MethodOrder -and -not (Test-Path -LiteralPath $ClipCheckpoint -PathType Leaf)) {
    throw "Missing TIR CLIP checkpoint: $ClipCheckpoint"
}
if ($Mode -eq 'smoke') {
    $McftEpochs = 1; $TirEpochs = 1; $McftBatch = 2; $TirBatch = 32
    $TirWarmupEpochs = 0
    $TirModalityWarmupEpochs = 0
    $TirTextUnfreezeEpoch = 1
}

$RunId = 'vds2raw_{0}_{1}_{2}' -f $Mode, (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID
$RunDir = Join-Path $OutputRoot $RunId
$ArtifactDir = Join-Path $RunDir 'artifacts'
$LogDir = Join-Path $RunDir 'logs'
$SummaryDir = Join-Path $RunDir 'summary'
$StatsFile = Join-Path $RunDir 'normalization_stats.json'
New-Item -ItemType Directory -Force -Path $ArtifactDir, $LogDir, $SummaryDir | Out-Null
Write-Host "Run directory: $RunDir"
Write-Host "Method order: $($MethodOrder -join ' ')"

& $PythonBin (Join-Path $RepoRoot 'vds2raw\audit.py') `
    --data-root $DataRoot --stats-file $StatsFile `
    --report-file (Join-Path $RunDir 'dataset_audit.json')
if ($LASTEXITCODE -ne 0) { throw 'VDS2Raw audit failed.' }

function Invoke-Method([string]$Tag, [string[]]$Arguments) {
    $log = Join-Path $LogDir "$Tag.log"
    Write-Host ">>> $Tag (log=$log)"
    & $PythonBin @Arguments 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "$Tag failed; log=$log" }
}

foreach ($seed in $Seeds) {
    $common = @(
        '--data-root', $DataRoot, '--save-dir', $ArtifactDir,
        '--seed', $seed,
        '--num-workers', (Env-OrDefault 'VDS2RAW_NUM_WORKERS' '0'), '--skip-audit'
    )
    $mcftSmokeArgs = if ($Mode -eq 'smoke') { @('--max-train-batches', '1') } else { @() }
    $tirSmokeArgs = @()
    foreach ($method in $MethodOrder) {
        switch ($method) {
            'mcft3' {
                $arguments = @(
                    (Join-Path $RepoRoot 'vds2raw\train_mcft.py'),
                    '--epochs', "$McftEpochs", '--batch-size', "$McftBatch",
                    '--lr', (Env-OrDefault 'VDS2RAW_MCFT_LR' '0.0005'),
                    '--weight-decay', (Env-OrDefault 'VDS2RAW_MCFT_WEIGHT_DECAY' '0.0001'),
                    '--step-size', (Env-OrDefault 'VDS2RAW_MCFT_STEP_SIZE' '20'),
                    '--gamma', (Env-OrDefault 'VDS2RAW_MCFT_GAMMA' '0.9'),
                    '--label-smoothing', (Env-OrDefault 'VDS2RAW_MCFT_LABEL_SMOOTHING' '0.05'),
                    '--loss-ita-weight', (Env-OrDefault 'VDS2RAW_MCFT_ITA_WEIGHT' '0.05'),
                    '--loss-itm-weight', (Env-OrDefault 'VDS2RAW_MCFT_ITM_WEIGHT' '1.0'),
                    '--ais-dropout-probability', (Env-OrDefault 'VDS2RAW_MCFT_AIS_DROPOUT' '0.3'),
                    '--early-stopping-patience', (Env-OrDefault 'VDS2RAW_MCFT_EARLY_STOPPING_PATIENCE' '25'),
                    '--early-stopping-min-epochs', (Env-OrDefault 'VDS2RAW_MCFT_EARLY_STOPPING_MIN_EPOCHS' '60'),
                    '--stats-file', $StatsFile
                ) + $common + $mcftSmokeArgs
                Invoke-Method -Tag "mcft3_seed$seed" -Arguments $arguments
            }
            'tir' {
                $arguments = @(
                    (Join-Path $RepoRoot 'vds2raw\train_tir.py'),
                    '--epochs', "$TirEpochs", '--batch-size', "$TirBatch",
                    '--eval-frequency', '1',
                    '--clip-checkpoint', $ClipCheckpoint,
                    '--lora-lr', (Env-OrDefault 'VDS2RAW_TIR_LORA_LR' '0.0002'),
                    '--text-lora-lr-scale', (Env-OrDefault 'VDS2RAW_TIR_TEXT_LORA_LR_SCALE' '0.25'),
                    '--prompt-lr', (Env-OrDefault 'VDS2RAW_TIR_PROMPT_LR' '0.0005'),
                    '--adapter-lr', (Env-OrDefault 'VDS2RAW_TIR_ADAPTER_LR' '0.001'),
                    '--gate-lr', (Env-OrDefault 'VDS2RAW_TIR_GATE_LR' '0.002'),
                    '--lora-weight-decay', (Env-OrDefault 'VDS2RAW_TIR_LORA_WEIGHT_DECAY' '0.01'),
                    '--weight-decay', (Env-OrDefault 'VDS2RAW_TIR_TASK_WEIGHT_DECAY' '0.0005'),
                    '--lora-rank', (Env-OrDefault 'VDS2RAW_TIR_LORA_RANK' '16'),
                    '--lora-alpha', (Env-OrDefault 'VDS2RAW_TIR_LORA_ALPHA' '8'),
                    '--lora-dropout', (Env-OrDefault 'VDS2RAW_TIR_LORA_DROPOUT' '0.25'),
                    '--lora-encoder', 'both', '--lora-params', 'q', 'k', 'v',
                    '--ais-dropout-probability', (Env-OrDefault 'VDS2RAW_TIR_AIS_DROPOUT' '0.1'),
                    '--nir-initial-scale', (Env-OrDefault 'VDS2RAW_TIR_NIR_INITIAL_SCALE' '0.05'),
                    '--ais-initial-scale', (Env-OrDefault 'VDS2RAW_TIR_AIS_INITIAL_SCALE' '0.75'),
                    '--modality-warmup-epochs', "$TirModalityWarmupEpochs",
                    '--text-unfreeze-epoch', "$TirTextUnfreezeEpoch",
                    '--nir-aux-loss-weight', (Env-OrDefault 'VDS2RAW_TIR_NIR_AUX_LOSS_WEIGHT' '0.2'),
                    '--ais-aux-loss-weight', (Env-OrDefault 'VDS2RAW_TIR_AIS_AUX_LOSS_WEIGHT' '0.2'),
                    '--tir-min-weight', (Env-OrDefault 'VDS2RAW_TIR_MIN_TOKEN_WEIGHT' '0.5'),
                    '--tir-max-weight', (Env-OrDefault 'VDS2RAW_TIR_MAX_TOKEN_WEIGHT' '1.5'),
                    '--class-weight-power', (Env-OrDefault 'VDS2RAW_TIR_CLASS_WEIGHT_POWER' '0.5'),
                    '--source-loss-weight', (Env-OrDefault 'VDS2RAW_TIR_SOURCE_LOSS_WEIGHT' '0.4'),
                    '--target-loss-weight', (Env-OrDefault 'VDS2RAW_TIR_TARGET_LOSS_WEIGHT' '0.6'),
                    '--label-smoothing', (Env-OrDefault 'VDS2RAW_TIR_LABEL_SMOOTHING' '0.0'),
                    '--scheduler', (Env-OrDefault 'VDS2RAW_TIR_SCHEDULER' 'cosine'),
                    '--warmup-epochs', "$TirWarmupEpochs",
                    '--scheduler-horizon-epochs', (Env-OrDefault 'VDS2RAW_TIR_SCHEDULER_HORIZON_EPOCHS' '100'),
                    '--min-lr-ratio', (Env-OrDefault 'VDS2RAW_TIR_MIN_LR_RATIO' '0.1'),
                    '--augment',
                    '--natural-train-sampling', '--minibatch-updates',
                    '--early-stopping-patience', (Env-OrDefault 'VDS2RAW_TIR_EARLY_STOPPING_PATIENCE' '0'),
                    '--early-stopping-min-epochs', (Env-OrDefault 'VDS2RAW_TIR_EARLY_STOPPING_MIN_EPOCHS' '60'),
                    '--stats-file', $StatsFile
                ) + $common + $tirSmokeArgs
                Invoke-Method -Tag "tir_seed$seed" -Arguments $arguments
            }
        }
    }
}

& $PythonBin (Join-Path $RepoRoot 'vds2raw\summarize.py') `
    --results-dir $ArtifactDir --output-dir $SummaryDir
if ($LASTEXITCODE -ne 0) { throw 'Result summary failed.' }
Write-Host "All VDS2Raw methods completed. Summary: $SummaryDir"
