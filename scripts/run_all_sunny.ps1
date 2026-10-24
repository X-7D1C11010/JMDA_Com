# =============================================================================
# One-click experiments: source = 晴天 (sunny), targets = 逆光/雨天/雾天/黑天
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_sunny.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_sunny.ps1 -Epochs 5  # 显式统一覆盖（冒烟）
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_sunny.ps1 -Methods CCSA,d-SNE
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_sunny.ps1 -Seeds 42 -Targets backlight -Methods PDA
#
# Every method keeps its OWN original supervision setting; methods are never
# mixed across protocols and each protocol is reported in its own summary table:
#
#   UDA (no target labels at all)
#     DANN, DeepCORAL, CDAN, MDD, MCC  - single-modal VIS, per original papers
#     PDA                              - CLIP ViT-B/16 prompt + pseudo labels
#   SDA (few-shot labelled target subset, paper main setting)
#     CCSA, d-SNE                      - both were released as supervised DA methods
#   Supervised (source-only, multimodal)
#     MCFT                             - VIS + IR contrastive fusion transformer
#
# Each target weather is an INDEPENDENT experiment. Each method x target runs
# once per seed in -Seeds (default 42..46) = 5 independent runs.
#
# Efficiency is recorded two ways, both into the same -OutDir:
#   * params / FLOPs / inference latency / throughput / GPU memory
#                  -> tools/benchmark_efficiency.py (micro-benchmark)
#   * per-epoch training time / peak GPU memory / total training time
#                  -> EfficiencyLogger inside each training script
# =============================================================================
param(
    [int]$Epochs = 0,                       # 0 = 各方法官方预算；>0 = 显式统一覆盖
    [string[]]$Seeds = @('42', '43', '44', '45', '46'),
    [string[]]$Targets = @('backlight', 'rainy', 'foggy', 'dark'),
    [string[]]$Methods = @(),                 # empty = all methods below
    [string]$Python = 'D:\Anaconda\envs\pytorch\python.exe',
    [string]$OutDir = '',
    [int]$BenchRepeats = 5,
    [switch]$SkipBenchmark,
    [int]$TargetLabelsPerClass = 10,          # SDA label budget (CCSA / d-SNE)
    [int]$SdaSeed = 42,                       # seed used by tools/make_sda_lists.py
    [string]$ClipCacheDir = '',               # PDA: CLIP weight cache ('' = ~/.cache/clip)
    [int]$DANNEpochs = 100,
    [int]$DeepCoralEpochs = 20,
    [int]$MDDEpochs = 20,
    [int]$MCCEpochs = 20,
    [int]$PDAEpochs = 20,
    [int]$CCSAEpochs = 80,
    [int]$DSNEEpochs = 20,
    [int]$MCFTEpochs = 500,
    [int]$CdanIterations = 100004
)

$ErrorActionPreference = 'Continue'

$Repo = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $Repo 'efficiency_results\all_sunny' }
$LogDir = Join-Path $OutDir 'logs'
New-Item -ItemType Directory -Force -Path $OutDir, $LogDir | Out-Null

# Method -> @{ script; protocol; args }  (args are appended per run)
$MethodSpecs = [ordered]@{
    'DANN'      = @{ script = 'DANN\train_dann.py';           protocol = 'UDA';        args = @() }
    'DeepCORAL' = @{ script = 'DeepCORAL\train_deepcoral.py'; protocol = 'UDA';        args = @() }
    'CDAN'      = @{ script = 'CDAN\train_cdan.py';           protocol = 'UDA';        args = @() }
    'MDD'       = @{ script = 'MDD\train_mdd.py';             protocol = 'UDA';        args = @() }
    'MCC'       = @{ script = 'MCC\train_mcc.py';             protocol = 'UDA';        args = @() }
    'PDA'       = @{ script = 'PDA\train_pda.py';             protocol = 'UDA';        args = @() }
    'CCSA'      = @{ script = 'CCSA\train_ccsa.py';           protocol = 'SDA';        args = @(
                        '--target-labels-per-class', $TargetLabelsPerClass,
                        '--sda-seed', $SdaSeed) }
    'd-SNE'     = @{ script = 'd-SNE\train_dsne.py';          protocol = 'SDA';        args = @(
                        '--target-labels-per-class', $TargetLabelsPerClass,
                        '--sda-seed', $SdaSeed) }
    'MCFT'      = @{ script = 'MCFT\train_mcft.py';           protocol = 'Supervised'; args = @() }
}

$MethodEpochs = [ordered]@{
    'DANN'      = $DANNEpochs
    'DeepCORAL' = $DeepCoralEpochs
    'MDD'       = $MDDEpochs
    'MCC'       = $MCCEpochs
    'PDA'       = $PDAEpochs
    'CCSA'      = $CCSAEpochs
    'd-SNE'     = $DSNEEpochs
    'MCFT'      = $MCFTEpochs
}
if ($Epochs -gt 0) {
    foreach ($name in @($MethodEpochs.Keys)) { $MethodEpochs[$name] = $Epochs }
}

# ``powershell -File script.ps1 -Methods CCSA,d-SNE`` binds "CCSA,d-SNE" as ONE
# string, so comma-separated single arguments are split here too.  ``-Methods
# CCSA,d-SNE`` (PowerShell host) and ``-Methods CCSA -Methods d-SNE`` both work.
function Expand-List([string[]]$Values) {
    $out = @()
    foreach ($v in $Values) {
        foreach ($part in ($v -split ',')) {
            $t = $part.Trim()
            if ($t) { $out += $t }
        }
    }
    return $out
}

$Methods = Expand-List $Methods
$Targets = Expand-List $Targets
$SeedList = @(Expand-List $Seeds | ForEach-Object { [int]$_ })
$Seeds = $SeedList

# Resolve the selection, keeping the declaration order above.
if (-not $Methods -or $Methods.Count -eq 0) {
    $Selected = @($MethodSpecs.Keys)
} else {
    $Selected = @()
    foreach ($m in $MethodSpecs.Keys) {
        if ($Methods -contains $m) { $Selected += $m }
    }
    $unknown = @($Methods | Where-Object { $MethodSpecs.Keys -notcontains $_ })
    if ($unknown.Count -gt 0) {
        Write-Host "Unknown method(s): $($unknown -join ', ')" -ForegroundColor Red
        Write-Host "Available: $($MethodSpecs.Keys -join ', ')" -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Path $Python)) {
    Write-Host "Python interpreter not found: $Python" -ForegroundColor Red
    exit 1
}

# SDA methods read a shared labelled subset generated by tools/make_sda_lists.py.
foreach ($m in @('CCSA', 'd-SNE')) {
    if ($Selected -contains $m) {
        foreach ($t in $Targets) {
            $listFile = Join-Path $Repo "data_lists\${t}_sda_${TargetLabelsPerClass}shot_seed${SdaSeed}.txt"
            if (-not (Test-Path $listFile)) {
                Write-Host "Missing SDA subset: $listFile" -ForegroundColor Red
                Write-Host "Generate it first: python tools\make_sda_lists.py" -ForegroundColor Yellow
                exit 1
            }
        }
    }
}

Write-Host "=============================================================="
Write-Host " Experiments  (source = sunny, target = other weathers)"
Write-Host "--------------------------------------------------------------"
Write-Host " Python   : $Python"
Write-Host (" Epochs   : DANN={0}, DeepCORAL={1}, MDD={2}, MCC={3}, PDA={4}, CCSA={5}, d-SNE={6}, MCFT={7}" -f `
    $MethodEpochs.DANN, $MethodEpochs.DeepCORAL, $MethodEpochs.MDD,
    $MethodEpochs.MCC, $MethodEpochs.PDA, $MethodEpochs.CCSA,
    $MethodEpochs.'d-SNE', $MethodEpochs.MCFT)
Write-Host " CDAN     : $CdanIterations iterations"
Write-Host " Seeds    : $($Seeds -join ', ')"
Write-Host " Targets  : $($Targets -join ', ')"
Write-Host " Methods  : $($Selected -join ', ')"
if ($Selected -contains 'CCSA' -or $Selected -contains 'd-SNE') {
    Write-Host " SDA      : ${TargetLabelsPerClass} labels/class, subset seed $SdaSeed (fixed for all runs)"
}
Write-Host " OutDir   : $OutDir"
Write-Host "=============================================================="

# ── 1) efficiency micro-benchmark (params / FLOPs / latency / memory) ────────
if (-not $SkipBenchmark) {
    Write-Host "`n[1/2] Efficiency micro-benchmark ..."
    $benchScript = Join-Path $Repo 'tools\benchmark_efficiency.py'

    # tools/benchmark_efficiency.py keys ADAPTERS in lowercase, so the display
    # names above must be mapped before they are passed to --methods.
    $AdapterKeys = @{
        'DANN' = 'dann'; 'DeepCORAL' = 'deepcoral'; 'CDAN' = 'cdan'
        'MDD' = 'mdd'; 'MCC' = 'mcc'; 'PDA' = 'pda'
        'CCSA' = 'ccsa'; 'd-SNE' = 'dsne'; 'MCFT' = 'mcft'
    }
    $benchGroups = [ordered]@{
        'UDA'        = @('DANN', 'DeepCORAL', 'CDAN', 'MDD', 'MCC', 'PDA')
        'SDA'        = @('CCSA', 'd-SNE')
        'Supervised' = @('MCFT')
    }
    foreach ($proto in $benchGroups.Keys) {
        $groupMethods = @($benchGroups[$proto] | Where-Object { $Selected -contains $_ })
        if ($groupMethods.Count -eq 0) { continue }
        $groupKeys = @($groupMethods | ForEach-Object { $AdapterKeys[$_] })
        $protoLog = Join-Path $LogDir "benchmark_$($proto.ToLower()).log"
        $cli = @('-B', $benchScript,
                 '--methods', (($groupKeys) -join ','),
                 '--protocol', $proto,
                 '--benchmark_repeats', $BenchRepeats,
                 '--efficiency_output_dir', $OutDir)
        if ($proto -eq 'SDA') {
            $cli += @('--target-labels-per-class', $TargetLabelsPerClass)
        }
        Write-Host "      [$proto] $($groupMethods -join ', ')"
        & $Python @cli *> $protoLog
        Write-Host ("      exit={0}   log={1}" -f $LASTEXITCODE, $protoLog)
    }
}

# ── 2) training runs: method x target x seed ────────────────────────────────
$checkpointDir = Join-Path $OutDir 'checkpoints'
New-Item -ItemType Directory -Force -Path $checkpointDir | Out-Null

$total = $Selected.Count * $Targets.Count * $Seeds.Count
$idx = 0
$failed = @()

foreach ($method in $Selected) {
    $spec = $MethodSpecs[$method]
    $scriptPath = Join-Path $Repo $spec['script']
    foreach ($target in $Targets) {
        foreach ($seed in $Seeds) {
            $idx++
            $tag = '{0}_{1}_seed{2}' -f $method, $target, $seed
            $log = Join-Path $LogDir "$tag.log"
            Write-Host ("[{0}/{1}] {2}  target={3}  seed={4}  protocol={5}" -f `
                        $idx, $total, $method, $target, $seed, $spec['protocol'])

            $cli = @('-B', $scriptPath,
                     '--target', $target,
                     '--seed', $seed,
                     '--efficiency_output_dir', $OutDir,
                     '--save-dir', $checkpointDir)
            if ($method -eq 'CDAN') {
                $cli += @('--num-iterations', $CdanIterations)
            } else {
                $cli += @('--epochs', $MethodEpochs[$method])
            }
            $cli += $spec['args']
            if ($method -eq 'PDA' -and $ClipCacheDir) {
                $cli += @('--clip-cache-dir', $ClipCacheDir)
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
