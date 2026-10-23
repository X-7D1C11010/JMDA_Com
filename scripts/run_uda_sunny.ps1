# =============================================================================
# UDA experiments: source = 晴天 (sunny), targets = 逆光/雨天/雾天/黑天
#
# Methods (each keeps its ORIGINAL single-modal / multi-modal setting):
#   DANN, DeepCORAL, CDAN, MDD, MCC, PDA -> all single-modal (VIS), per their own
#   original definitions. No target labels are used during training (UDA).
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
    [string]$Python = 'D:\Anaconda\envs\pytorch\python.exe',
    [string]$OutDir = '',
    [string]$DataDir = '',
    [int]$CdanIterations = 100004,
    [int]$DANNEpochs = 100,
    [int]$DeepCoralEpochs = 20,
    [int]$MDDEpochs = 20,
    [int]$MCCEpochs = 20,
    [int]$PDAEpochs = 20,
    [string]$ClassNamesFile = $env:JMDA_CLASS_NAMES_FILE,
    [string]$ClipCacheDir = '',
    [int]$BenchRepeats = 5,
    [switch]$SkipBenchmark
)

$ErrorActionPreference = 'Continue'

$Repo = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $Repo 'efficiency_results\uda_sunny' }
if (-not $DataDir) { $DataDir = Join-Path $Repo 'data_lists_vis' }
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
if (-not $ClassNamesFile -or -not (Test-Path -LiteralPath $ClassNamesFile -PathType Leaf)) {
    Write-Host 'PDA requires a real ordered 14-class English name file. Pass -ClassNamesFile or set JMDA_CLASS_NAMES_FILE.' -ForegroundColor Red
    exit 1
}

Write-Host "=============================================================="
Write-Host " UDA experiments  (source = sunny, target = other weathers)"
Write-Host "--------------------------------------------------------------"
Write-Host " Python  : $Python"
Write-Host " Epochs  : DANN=$($MethodEpochs.DANN), DeepCORAL=$($MethodEpochs.DeepCORAL), MDD=$($MethodEpochs.MDD), MCC=$($MethodEpochs.MCC), PDA=$($MethodEpochs.PDA)"
Write-Host " CDAN iters: $CdanIterations"
Write-Host " Seeds   : $($Seeds -join ', ')"
Write-Host " Targets : $($Targets -join ', ')"
Write-Host " Methods : $($Methods.Keys -join ', ')"
Write-Host " OutDir  : $OutDir"
Write-Host "=============================================================="

# ── 1) efficiency micro-benchmark (params / FLOPs / latency / memory) ────────
if (-not $SkipBenchmark) {
    Write-Host "`n[1/2] Efficiency micro-benchmark ..."
    $benchLog = Join-Path $LogDir 'benchmark.log'
    $benchScript = Join-Path $Repo 'tools\benchmark_efficiency.py'
    & $Python -B $benchScript `
        --methods 'dann,deepcoral,cdan,mdd,mcc,pda' `
        --protocol UDA `
        --benchmark_repeats $BenchRepeats `
        --efficiency_output_dir $OutDir *> $benchLog
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

            # CDAN 按官方配置为迭代制（num_iterations=100004），不用 --epochs。
            if ($method -eq 'CDAN') {
                & $Python -B $scriptPath `
                    --target $target `
                    --num-iterations $CdanIterations `
                    --seed $seed `
                    --data-dir $DataDir `
                    --efficiency_output_dir $OutDir `
                    --save-dir $checkpointDir *> $log
            } elseif ($method -eq 'PDA') {
                & $Python -B $scriptPath `
                    --target $target `
                    --epochs $MethodEpochs[$method] `
                    --seed $seed `
                    --data-dir $DataDir `
                    --efficiency_output_dir $OutDir `
                    --save-dir $checkpointDir `
                    --class-names-file $ClassNamesFile `
                    --clip-cache-dir $ClipCacheDir *> $log
            } else {
                & $Python -B $scriptPath `
                    --target $target `
                    --epochs $MethodEpochs[$method] `
                    --seed $seed `
                    --data-dir $DataDir `
                    --efficiency_output_dir $OutDir `
                    --save-dir $checkpointDir *> $log
            }

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
& $Python -B -c $pyCode

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
