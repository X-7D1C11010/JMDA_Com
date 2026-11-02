# =============================================================================
# UDA experiments: source = 晴天 (sunny), targets = 逆光/雨天/雾天/黑天
#
# Without -Multimodal the historical VIS protocol is retained. With
# -Multimodal, all enabled methods consume exact-paired VIS+IR observations and
# use the project multimodal extensions documented in MULTIMODAL_UDA_PROTOCOL.md.
#
# Per target weather the experiment is INDEPENDENT. Each method x target is run
# for every seed in -Seeds (default 42..46), i.e. 5 independent runs.
#
# Efficiency is recorded two ways, both written into the SAME -OutDir so the
# final summary table (efficiency_summary.csv / efficiency_table.md) holds:
#   * params / FLOPs / inference latency / throughput / GPU memory
#                       -> tools/benchmark_efficiency.py (micro-benchmark)
#   * per-epoch training time / peak GPU memory / total train time
#                       -> EfficiencyLogger inside each training script
#
# Usage (run from the repo root or anywhere):
#   powershell -ExecutionPolicy Bypass -File scripts\run_uda_sunny.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_uda_sunny.ps1 -Epochs 5  # 冒烟：覆盖各方法轮数
#   powershell -ExecutionPolicy Bypass -File scripts\run_uda_sunny.ps1 -Seeds 42 -Targets backlight   # quick trial
# =============================================================================
param(
    [int]$Epochs = 0,
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [string[]]$Targets = @('backlight', 'rainy', 'foggy', 'dark'),
    [string[]]$IncludeMethods = @(),
    [string]$Python = 'D:\Anaconda\envs\pytorch\python.exe',
    [string]$OutDir = '',
    [string]$DataDir = '',
    [int]$CdanIterations = 2000,
    [int]$DANNEpochs = 100,
    [int]$DeepCoralEpochs = 20,
    [int]$MDDEpochs = 20,
    [int]$MCCEpochs = 20,
    [int]$PDAEpochs = 20,
    [string]$ClassNamesFile = $env:JMDA_CLASS_NAMES_FILE,
    [string]$ClipCacheDir = '',
    [string]$DeepCoralWeights = '',
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 0,
    [int]$PrefetchFactor = 4,
    [int]$WorkerTimeoutSeconds = 300,
    [int]$WindowsWorkerCap = 2,
    [int]$BenchRepeats = 5,
    [string]$ImageBackbone = '',
    [int]$ImageBatchSize = 16,
    [switch]$EnhancedAugmentation,
    [switch]$Multimodal,
    [switch]$IncludePausedMethods,
    [int]$EarlyStoppingPatience = 0,
    [double]$EarlyStoppingMinDelta = 0.0,
    [switch]$SkipBenchmark
)

$ErrorActionPreference = 'Continue'

$Repo = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $Repo 'efficiency_results\uda_sunny' }
if (-not $DataDir) { $DataDir = Join-Path $Repo 'data_lists_vis' }
if (-not $ClassNamesFile) { $ClassNamesFile = Join-Path $Repo 'class.txt' }
if (-not $ClipCacheDir) { $ClipCacheDir = Join-Path $Repo 'pretrained_weights\clip' }
if (-not $DeepCoralWeights) {
    $candidate = Join-Path $Repo 'pretrained_weights\deepcoral_alexnet_owt_4df8aa71.pth'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { $DeepCoralWeights = $candidate }
}
$env:TORCH_HOME = Join-Path $Repo 'pretrained_weights\torch'
$LogDir = Join-Path $OutDir 'logs'
New-Item -ItemType Directory -Force -Path $OutDir, $LogDir | Out-Null

# Method name -> training script (relative to repo root)
$Methods = [ordered]@{
    'DANN'      = 'DANN\train_dann.py'
    'DeepCORAL' = 'DeepCORAL\train_deepcoral.py'
    'CDAN'      = 'CDAN\train_cdan.py'
    'MDD'       = 'MDD\train_mdd.py'
    'MCC'       = 'MCC\train_mcc.py'
    'PDA'       = 'PDA\train_pda.py'
}
$PausedMethods = @('MDD', 'MCC')
if (-not $IncludePausedMethods) {
    foreach ($paused in $PausedMethods) { $Methods.Remove($paused) }
}
if ($IncludeMethods.Count -gt 0) {
    $selected = [ordered]@{}
    foreach ($requested in $IncludeMethods) {
        $canonical = @($Methods.Keys | Where-Object { $_ -ieq $requested })
        if ($canonical.Count -ne 1) {
            $pauseHint = if ($PausedMethods -contains $requested) {
                ' (temporarily paused; add -IncludePausedMethods to override)'
            } else { '' }
            Write-Host ("Unknown or disabled method '{0}'{1}. Available methods: {2}" -f `
                       $requested, $pauseHint, ($Methods.Keys -join ', ')) -ForegroundColor Red
            exit 1
        }
        $name = $canonical[0]
        $selected[$name] = $Methods[$name]
    }
    $Methods = $selected
}

# 默认使用每种方法发布配方中的训练预算。-Epochs > 0 仅用于用户明确要求的
# 统一冒烟预算；CDAN 始终使用自己的迭代制参数。
$MethodEpochs = [ordered]@{
    'DANN'      = $DANNEpochs
    'DeepCORAL' = $DeepCoralEpochs
    'MDD'       = $MDDEpochs
    'MCC'       = $MCCEpochs
    'PDA'       = $PDAEpochs
}
if ($Epochs -gt 0) {
    foreach ($name in @($MethodEpochs.Keys)) { $MethodEpochs[$name] = $Epochs }
}

if (-not (Test-Path $Python)) {
    Write-Host "Python interpreter not found: $Python" -ForegroundColor Red
    exit 1
}
if ($NumWorkers -lt 0 -or $EvalNumWorkers -lt 0 -or
    $PrefetchFactor -lt 1 -or $WorkerTimeoutSeconds -lt 1) {
    Write-Host 'Invalid DataLoader settings.' -ForegroundColor Red
    exit 1
}
if ($ImageBackbone -and $ImageBackbone -ine 'ResNet50') {
    Write-Host 'ImageBackbone currently supports only ResNet50 or an empty value.' -ForegroundColor Red
    exit 1
}
if ($ImageBatchSize -lt 2) {
    Write-Host 'ImageBatchSize must be >= 2.' -ForegroundColor Red
    exit 1
}
if ($EarlyStoppingPatience -lt 0 -or $EarlyStoppingMinDelta -lt 0) {
    Write-Host 'Early stopping patience/min-delta must be non-negative.' -ForegroundColor Red
    exit 1
}
if ($EarlyStoppingPatience -gt 0) {
    Write-Host 'Target-validation early stopping/model selection is disabled by this protocol.' -ForegroundColor Red
    exit 1
}
if ($Multimodal -and -not $SkipBenchmark) {
    Write-Host 'Multimodal micro-benchmark is not yet protocol-matched; skipping it to avoid recording single-modal complexity under VIS+IR runs.' -ForegroundColor Yellow
    $SkipBenchmark = $true
}

# Windows uses multiprocessing spawn. Every worker imports PyTorch and CUDA
# DLLs, so two domain loaders with four workers each can consume over 10 GiB of
# committed host memory. Cap only the local Windows launcher; Linux/server
# launchers keep their independently configured worker budget.
$EffectiveNumWorkers = $NumWorkers
$UsePersistentWorkers = $true
if ($env:OS -eq 'Windows_NT') {
    if ($WindowsWorkerCap -lt 0) {
        Write-Host 'WindowsWorkerCap must be >= 0.' -ForegroundColor Red
        exit 1
    }
    if ($EffectiveNumWorkers -gt $WindowsWorkerCap) {
        Write-Host ("Windows safety: capping training workers from {0} to {1}." -f `
                   $EffectiveNumWorkers, $WindowsWorkerCap) -ForegroundColor Yellow
        $EffectiveNumWorkers = $WindowsWorkerCap
    }
    # With only two workers per training domain and zero evaluation workers,
    # persistence is safe and avoids repeatedly spawning PyTorch for the
    # cycling loaders used by CDAN/MDD/MCC.
    $UsePersistentWorkers = ($EffectiveNumWorkers -gt 0)
}
$env:PYTHONUNBUFFERED = '1'
if (-not $ClassNamesFile -or -not (Test-Path -LiteralPath $ClassNamesFile -PathType Leaf)) {
    Write-Host 'PDA requires a real ordered 14-class English name file. Pass -ClassNamesFile or set JMDA_CLASS_NAMES_FILE.' -ForegroundColor Red
    exit 1
}

Write-Host "=============================================================="
Write-Host " UDA experiments  (source = sunny, target = other weathers)"
Write-Host "--------------------------------------------------------------"
Write-Host " Python  : $Python"
$epochSummary = @($Methods.Keys | Where-Object { $_ -ne 'CDAN' } |
    ForEach-Object { "$_=$($MethodEpochs[$_])" }) -join ', '
Write-Host " Epochs  : $epochSummary"
Write-Host " CDAN iters: $CdanIterations"
Write-Host " Seeds   : $($Seeds -join ', ')"
Write-Host " Targets : $($Targets -join ', ')"
Write-Host " Methods : $($Methods.Keys -join ', ')"
Write-Host " Paused  : MDD, MCC$(if ($IncludePausedMethods) { ' (override enabled)' } else { '' })"
Write-Host " OutDir  : $OutDir"
Write-Host ((" Loader  : train_workers={0}, eval_workers={1}, prefetch={2}, " +
             "persistent={3}, timeout={4}s") -f $EffectiveNumWorkers,
             $EvalNumWorkers, $PrefetchFactor, $UsePersistentWorkers,
             $WorkerTimeoutSeconds)
Write-Host " TORCH_HOME: $env:TORCH_HOME"
Write-Host " Image backbone override: $(if ($ImageBackbone) { $ImageBackbone } else { 'method default' })"
Write-Host " Image backbone batch size: $ImageBatchSize"
Write-Host " Enhanced augmentation: $EnhancedAugmentation"
Write-Host " Modalities: $(if ($Multimodal) { 'VIS+IR exact pairs' } else { 'VIS' })"
Write-Host " Early stopping: patience=$EarlyStoppingPatience, min_delta=$EarlyStoppingMinDelta"
if ($EarlyStoppingPatience -gt 0) {
    Write-Host ' WARNING: early stopping uses labelled target validation accuracy; protocol becomes target-validation-selected UDA.' -ForegroundColor Yellow
}
Write-Host "=============================================================="

# ── 1) efficiency micro-benchmark (params / FLOPs / latency / memory) ────────
if (-not $SkipBenchmark) {
    Write-Host "`n[1/2] Efficiency micro-benchmark ..."
    $benchLog = Join-Path $LogDir 'benchmark.log'
    $benchScript = Join-Path $Repo 'tools\benchmark_efficiency.py'
    $benchmarkMethods = (($Methods.Keys | ForEach-Object { $_.ToLowerInvariant() }) -join ',')
    $benchCli = @('-B', '-u', $benchScript,
        '--methods', $benchmarkMethods,
        '--protocol', 'UDA',
        '--benchmark_repeats', $BenchRepeats,
        '--efficiency_output_dir', $OutDir)
    if ($ImageBackbone) { $benchCli += @('--image-backbone', 'resnet50') }
    & $Python @benchCli `
        *> $benchLog
    Write-Host ("      benchmark exit={0}   log={1}" -f $LASTEXITCODE, $benchLog)
}

# ── 2) training runs: method x target x seed ────────────────────────────────
$checkpointDir = Join-Path $OutDir 'checkpoints'
New-Item -ItemType Directory -Force -Path $checkpointDir | Out-Null

$total = $Methods.Count * $Targets.Count * $Seeds.Count
$idx = 0
$failed = @()

foreach ($method in $Methods.Keys) {
    $scriptPath = Join-Path $Repo $Methods[$method]
    foreach ($target in $Targets) {
        foreach ($seed in $Seeds) {
            $idx++
            $tag = '{0}_{1}_seed{2}' -f $method, $target, $seed
            $log = Join-Path $LogDir "$tag.log"
            Write-Host ("[{0}/{1}] {2}  target={3}  seed={4}" -f $idx, $total, $method, $target, $seed)

            # CDAN is iteration-based; the project multimodal budget is 2000.
            $cli = @('-B', '-u', $scriptPath,
                     '--target', $target,
                     '--seed', $seed,
                     '--data-dir', $DataDir,
                     '--efficiency_output_dir', $OutDir,
                     '--save-dir', $checkpointDir,
                     '--num-workers', $EffectiveNumWorkers,
                     '--prefetch-factor', $PrefetchFactor,
                     '--pin-memory')
            if ($UsePersistentWorkers) {
                $cli += '--persistent-workers'
            } else {
                $cli += '--no-persistent-workers'
            }
            $cli += @('--eval-num-workers', $EvalNumWorkers,
                      '--worker-timeout-seconds', $WorkerTimeoutSeconds)
            if ($method -eq 'CDAN') {
                $cli += @('--num-iterations', $CdanIterations)
            } else {
                $cli += @('--epochs', $MethodEpochs[$method])
            }
            if ($method -eq 'PDA') {
                $cli += @('--class-names-file', $ClassNamesFile,
                          '--clip-cache-dir', $ClipCacheDir)
                if ($ImageBackbone) {
                    $cli += @('--batch-size', $ImageBatchSize,
                              '--test-batch-size', $ImageBatchSize)
                }
            }
            if ($method -eq 'DeepCORAL' -and $DeepCoralWeights) {
                $cli += @('--alexnet-weights', $DeepCoralWeights)
            }
            if ($EnhancedAugmentation) {
                $cli += '--enhanced-augmentation'
            }
            if ($Multimodal) {
                $cli += '--multimodal'
            }
            if ($EarlyStoppingPatience -gt 0) {
                $cli += @('--early-stop-patience', $EarlyStoppingPatience,
                          '--early-stop-min-delta', $EarlyStoppingMinDelta)
            }
            if ($ImageBackbone) {
                if ($method -eq 'DANN' -or $method -eq 'DeepCORAL') {
                    $cli += @('--backbone', 'resnet50')
                } elseif ($method -eq 'CDAN') {
                    $cli += @('--backbone', 'ResNet50')
                } elseif ($method -eq 'MDD' -or $method -eq 'MCC') {
                    $cli += @('--backbone', 'resnet50')
                }
                if ($method -eq 'DeepCORAL') {
                    $cli += @('--source-batch-size', $ImageBatchSize,
                              '--target-batch-size', $ImageBatchSize,
                              '--test-batch-size', $ImageBatchSize)
                } elseif ($method -ne 'PDA') {
                    $cli += @('--batch-size', $ImageBatchSize)
                }
            }
            & $Python @cli *> $log

            if ($LASTEXITCODE -ne 0) {
                Write-Host ("      FAILED (exit={0})  log={1}" -f $LASTEXITCODE, $log) -ForegroundColor Red
                $failed += $tag
            } else {
                Write-Host "      ok"
            }
        }
    }
}

# ── 3) regenerate combined summary (training + benchmark rows) ──────────────
Write-Host "`n[2/2] Regenerating summary ..."
$pyCode = "import sys; sys.path.insert(0, r'$Repo\tools'); " +
          "from efficiency_common import generate_summary_files; " +
          "generate_summary_files(r'$OutDir')"
& $Python -B -u -c $pyCode

Write-Host "`n=============================================================="
Write-Host " Done."
Write-Host " Raw runs CSV : $(Join-Path $OutDir 'efficiency_runs.csv')"
Write-Host " Summary CSV  : $(Join-Path $OutDir 'efficiency_summary.csv')"
Write-Host " Summary MD   : $(Join-Path $OutDir 'efficiency_table.md')"
Write-Host " Per-run logs : $LogDir"
if ($failed.Count -gt 0) {
    Write-Host (" Failed runs ({0}): {1}" -f $failed.Count, ($failed -join ', ')) -ForegroundColor Red
}
Write-Host "=============================================================="
