"""Create explicitly synthetic learning-curve logs, never real experiment results.

Source/target seed42 histories supply illustrative shapes only. Requested final
accuracies are assumptions supplied by the user, not predictions from a model.
No training, model loading, target-train labels or official summaries are used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("dann", "deepcoral", "cdan", "mdd", "mcc")
WEATHERS = ("dark", "foggy")
LABEL = "SIMULATED SCENARIO - NOT REAL EXPERIMENTAL RESULTS"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_references():
    index_path = ROOT/"results/current_local_student_peak_scan_20261115.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    references = {}
    for item in index["rows"]:
        if item["target"] not in WEATHERS or item["method"] not in METHODS:
            continue
        result_path = ROOT/item["file"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("simulated") or result.get("smoke") or result["seed"] != 42:
            raise ValueError("A reference must be a recorded, non-smoke seed42 run")
        directory = result_path.parent
        evaluation = [json.loads(s) for s in (directory/"evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
        training = [json.loads(s) for s in (directory/"training.jsonl").read_text(encoding="utf-8").splitlines()]
        source = [r for r in evaluation if r["partition"] == "source_train"]
        target = [r for r in evaluation if r["partition"] == "target_val"]
        epochs = [r["epoch"] for r in source]
        if epochs != [r["epoch"] for r in target] or epochs != [r["epoch"] for r in training]:
            raise ValueError("Reference epoch histories are not aligned")
        references[item["target"], item["method"]] = {
            "source_acc": np.array([r["metrics"]["accuracy"] for r in source]),
            "source_ce": np.array([r["source_ce"] for r in training]),
            "target_acc": np.array([r["metrics"]["accuracy"] for r in target]),
            "source_counts": np.array(source[0]["metrics"]["confusion_matrix"]).sum(1),
            "target_counts": np.array(target[0]["metrics"]["confusion_matrix"]).sum(1),
            "source_errors": np.array(source[-1]["metrics"]["confusion_matrix"]),
            "target_errors": np.array(result["target_val"]["confusion_matrix"]),
            "provenance": {
                "reference_seed": 42, "recorded_epochs": len(epochs),
                "result_file": result_path.relative_to(ROOT).as_posix(),
                "result_sha256": digest(result_path),
                "source_and_target_history_sha256": digest(directory/"evaluation.jsonl"),
                "source_training_history_sha256": digest(directory/"training.jsonl"),
                "recorded_target_peak_accuracy": result["target_val"]["accuracy"],
                "reference_was_selected_by_historical_peak": True,
                "not_an_unbiased_forecast_baseline": True,
            },
        }
    if set(references) != {(w, m) for w in WEATHERS for m in METHODS}:
        raise ValueError("Missing real reference histories")
    return references


def resample(values, epochs):
    return np.interp(np.linspace(0, 1, epochs), np.linspace(0, 1, len(values)), values)


def smooth(values, width=5):
    if width == 1:
        return values.copy()
    return np.convolve(np.pad(values, width//2, mode="edge"), np.ones(width)/width, mode="valid")


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    diagonal = matrix.diagonal().astype(float)
    true, predicted = matrix.sum(1), matrix.sum(0)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
    recall = np.divide(diagonal, true, out=np.zeros_like(diagonal), where=true > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(diagonal), where=precision+recall > 0)
    return {"accuracy": float(diagonal.sum()/matrix.sum()), "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()), "f1_macro": float(f1.mean()),
            "synthetic_samples": int(matrix.sum()), "synthetic_confusion_matrix": matrix.tolist()}


def synthetic_confusion(counts, accuracy, reference_errors, rng):
    # Correct-count construction makes all four metrics agree with integer
    # sample counts. No synthetic per-image predictions are claimed or saved.
    counts = np.asarray(counts, dtype=np.int64)
    labels = np.repeat(np.arange(len(counts)), counts)
    correct_total = int(np.clip(round(float(accuracy)*len(labels)), 0, len(labels)))
    chosen = rng.choice(len(labels), correct_total, replace=False)
    correct = np.bincount(labels[chosen], minlength=len(counts))
    matrix = np.zeros((len(counts), len(counts)), dtype=np.int64)
    for i, total in enumerate(counts):
        errors = reference_errors[i].astype(float)+.5
        errors[i] = 0
        errors /= errors.sum()
        matrix[i] = rng.multinomial(int(total-correct[i]), errors)
        matrix[i, i] = correct[i]
    assert np.array_equal(matrix.sum(1), counts)
    return metrics_from_confusion(matrix)


def correlated_noise(rng, epochs, scale):
    result = np.zeros(epochs)
    for i in range(1, epochs):
        result[i] = .8*result[i-1]+rng.normal(0, scale)
    # A zero end-point disturbance keeps the requested final-level assumption
    # separate from the explicitly recorded seed-specific endpoint offset.
    return result-result[-1]*np.linspace(0, 1, epochs)


def simulate_run(reference, weather, method, seed, epochs, nominal_endpoint):
    rng = np.random.default_rng(np.random.SeedSequence([seed, WEATHERS.index(weather), METHODS.index(method), 20261115]))
    time = np.linspace(0, 1, epochs)
    ref_source = resample(reference["source_acc"], epochs)
    ref_target = resample(reference["target_acc"], epochs)
    source_seed_offset = float(np.clip(rng.normal(0, .004), -.008, .008))
    simulated_source = np.clip(ref_source+source_seed_offset+correlated_noise(rng, epochs, .002), .01, 1)
    envelope = np.maximum.accumulate(smooth(ref_source))
    source_progress = (envelope-envelope[0])/max(envelope[-1]-envelope[0], .01)
    source_progress = np.clip(source_progress, 0, 1)
    clock_progress = (1-np.exp(-4*time))/(1-np.exp(-4))
    progress = .6*source_progress+.4*clock_progress
    progress[0], progress[-1] = 0., 1.
    seed_endpoint_offset = float(np.clip(rng.normal(0, .008), -.016, .016))
    endpoint = float(np.clip(nominal_endpoint+seed_endpoint_offset, .05, .98))
    amplitude = {"dann": .012, "deepcoral": .004, "cdan": .010, "mdd": .010, "mcc": .006}[method]
    phase = rng.uniform(0, 2*np.pi)
    oscillation = amplitude*np.sin(2*np.pi*4*time+phase)*np.sin(np.pi*time)
    target = np.clip((1-progress)*smooth(ref_target)+progress*endpoint
                     +oscillation+correlated_noise(rng, epochs, amplitude/4), .005, .995)
    target[-1] = endpoint
    ce = np.maximum(resample(reference["source_ce"], epochs)*(1+correlated_noise(rng, epochs, .025)), 1e-5)
    records = []
    for i in range(epochs):
        records.append({"simulated": True, "not_a_real_experiment": True, "artifact_label": LABEL,
            "record_type": "synthetic_learning_curve", "weather": weather, "method": method, "seed": seed,
            "epoch": i+1, "simulation_epochs": epochs, "reference_source_seed": 42,
            "reference_source_accuracy_interpolated": float(ref_source[i]),
            "reference_target_accuracy_interpolated": float(ref_target[i]),
            "simulated_source_train_metrics": synthetic_confusion(reference["source_counts"], simulated_source[i], reference["source_errors"], rng),
            "simulated_source_ce": float(ce[i]),
            "simulated_target_val_metrics": synthetic_confusion(reference["target_counts"], target[i], reference["target_errors"], rng),
            "nominal_target_final_accuracy": nominal_endpoint, "seed_target_final_accuracy_assumption": endpoint,
            "target_endpoint_is_user_assumption": True, "source_only_accuracy_forecast": False,
            "wall_clock_training_performed": False})
    return records


def render_curves(output, runs, endpoint_levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    details, panels = plt.subplots(2, 5, figsize=(17, 7), sharex=True, sharey=True)
    for wi, weather in enumerate(WEATHERS):
        for mi, method in enumerate(METHODS):
            group = [r for r in runs if r["weather"] == weather and r["method"] == method]
            target = np.array([[p["simulated_target_val_metrics"]["accuracy"]*100 for p in r["records"]] for r in group])
            source = np.array([[p["simulated_source_train_metrics"]["accuracy"]*100 for p in r["records"]] for r in group])
            x = np.arange(1, target.shape[1]+1)
            ax = axes[wi]
            ax.plot(x, target.mean(0), label=method.upper(), color=colors[method], linewidth=1.7)
            ax.fill_between(x, target.min(0), target.max(0), color=colors[method], alpha=.10)
            panel = panels[wi, mi]
            for curve in target:
                panel.plot(x, curve, color=colors[method], alpha=.22, linewidth=.7)
            panel.plot(x, target.mean(0), color=colors[method], label="Synthetic target", linewidth=1.7)
            panel.plot(x, source.mean(0), color="#475569", linestyle="--", label="Synthetic source")
            panel.set_title(f"{weather.upper()} / {method.upper()}", fontsize=10)
            panel.grid(alpha=.18); panel.set_ylim(0, 102)
            panel.set_xlabel("Illustrative epoch")
            if mi == 0:
                panel.set_ylabel("Simulated ACC (%)")
        level = 100*endpoint_levels[weather]
        axes[wi].axhline(level, color="black", linestyle=":", linewidth=.8, label=f"User-set level {level:g}%")
        axes[wi].set_title(weather.upper()+" - synthetic seeds42-46")
        axes[wi].set_xlabel("Illustrative epoch"); axes[wi].set_ylim(0, 102); axes[wi].grid(alpha=.18)
        axes[wi].legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Simulated target ACC (%)")
    for figure in [fig, details]:
        figure.suptitle(LABEL, color="#b91c1c", fontsize=14, weight="bold")
        figure.text(.5, .012, "Final levels are user assumptions, not forecasts. Bands: range of 5 synthetic seeds, not confidence intervals.",
                    ha="center", fontsize=8, color="#b91c1c")
        figure.tight_layout(rect=[0, .045, 1, .94])
    panels[0, 0].legend(fontsize=7)
    fig.savefig(output/"SIMULATED_target_accuracy_curves.png", dpi=160)
    details.savefig(output/"SIMULATED_source_target_curves.png", dpi=150)
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"simulations/weather_dark_foggy_seed42_46")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    parser.add_argument("--dark-final", type=float, default=.85)
    parser.add_argument("--foggy-final", type=float, default=.72)
    args = parser.parse_args()
    if args.epochs < 2 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Invalid epoch/seed specification")
    if not all(.05 < p < .98 for p in [args.dark_final, args.foggy_final]):
        parser.error("Assumed final accuracies must be fractions in (.05,.98)")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/"simulations"):
        parser.error("Synthetic logs must remain under the separate simulations directory")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite simulation artifacts: {output}")
    references = load_references()
    output.mkdir(parents=True)
    all_runs, summaries = [], []
    for weather in WEATHERS:
        level = args.dark_final if weather == "dark" else args.foggy_final
        peaks = np.array([references[weather, m]["provenance"]["recorded_target_peak_accuracy"] for m in METHODS])
        # Small method offsets retain only an illustrative relative ordering;
        # they are deliberately not estimates of new target-domain ability.
        offsets = np.clip((peaks-peaks.mean())*.15, -.018, .018)
        for mi, method in enumerate(METHODS):
            reference = references[weather, method]
            folder = output/weather/method
            folder.mkdir(parents=True)
            for seed in args.seeds:
                records = simulate_run(reference, weather, method, seed, args.epochs, level+float(offsets[mi]))
                stem = f"SIMULATED_{method}_{weather}_seed{seed}"
                with (folder/(stem+".jsonl")).open("w", encoding="utf-8") as f:
                    for row in records:
                        f.write(json.dumps(row, ensure_ascii=False)+"\n")
                lines = [LABEL, "User-specified endpoint scenario. No actual training was performed."]
                for row in records:
                    s, t = row["simulated_source_train_metrics"], row["simulated_target_val_metrics"]
                    lines.append(f"[SIMULATED] method={method} weather={weather} seed={seed} epoch={row['epoch']}/{args.epochs} "
                        f"source_ACC={100*s['accuracy']:.2f}% source_CE={row['simulated_source_ce']:.5f} "
                        f"target_ACC={100*t['accuracy']:.2f}% target_P_macro={100*t['precision_macro']:.2f}% "
                        f"target_R_macro={100*t['recall_macro']:.2f}% target_F1_macro={100*t['f1_macro']:.2f}%")
                (folder/(stem+".log")).write_text("\n".join(lines)+"\n", encoding="utf-8")
                peak = max(records, key=lambda r: (r["simulated_target_val_metrics"]["accuracy"], r["simulated_target_val_metrics"]["f1_macro"]))
                summaries.append({"simulated": True, "artifact_label": LABEL, "weather": weather, "method": method, "seed": seed,
                    "synthetic_final_epoch": args.epochs, "synthetic_final_metrics": records[-1]["simulated_target_val_metrics"],
                    "synthetic_peak_epoch": peak["epoch"], "synthetic_peak_metrics": peak["simulated_target_val_metrics"],
                    "requested_weather_endpoint": level, "assumed_method_endpoint": level+float(offsets[mi]),
                    "reference": reference["provenance"]})
                all_runs.append({"weather": weather, "method": method, "seed": seed, "records": records})
    manifest = {"simulated": True, "not_a_real_experiment": True, "artifact_label": LABEL,
        "created_UTC": datetime.now(timezone.utc).isoformat(), "synthetic_run_count": len(summaries),
        "epochs": args.epochs, "seeds": args.seeds, "user_assumed_final_accuracy": {"dark": args.dark_final, "foggy": args.foggy_final},
        "measured_multiseed_performance": False, "forecast_claim": False, "target_train_labels_read": False,
        "formula": "source progress 0.6 + synthetic epoch clock 0.4; interpolate recorded target shape toward USER-ASSUMED final level; AR noise + method oscillation",
        "source_history_epoch_mapping": "normalized interpolation to requested illustration length; not new observed source epochs",
        "method_offset_rule": "0.15 * (historical target peak - weather mean peak), clipped to +/-0.018; illustrative only",
        "seed_endpoint_offset_rule": "Normal(0,0.008), clipped to +/-0.016; arbitrary scenario spread, not empirical uncertainty",
        "confidence_intervals_computed": False, "real_result_directories_written": False,
        "metrics_rule": "integer synthetic confusion matrices with fixed reference class support; all four metrics recomputed consistently",
        "runs": summaries}
    (output/"SIMULATED_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (output/"README_SIMULATION_ONLY.md").write_text(
        "# 模拟情景日志：非真实实验\n\n"
        "**全部日志和曲线均为合成数据，不能计入真实训练成绩或用于证明方法性能。**\n\n"
        f"用户指定黑天末轮约{args.dark_final:.0%}、雾天约{args.foggy_final:.0%}；这两个终点不是源域结果推算出的预测。\n"
        f"共五方法×两天气×{len(args.seeds)}个合成seed，每项{args.epochs}轮。原始seed42源/目标曲线只提供形状参考，"
        "并按相对训练进度重采样；没有真实训练这些轮次，也没有真实运行seed43–46。\n\n"
        "每行都含simulated标记。人读日志为SIMULATED_*.log，结构化日志为SIMULATED_*.jsonl。"
        "所有指标由合成混淆矩阵一致计算；最终轮次与最高轮次分别记录，不能混用。"
        "方法差异和seed波动是人为指定的情景假设；图中阴影是五条合成曲线的范围，不是统计置信区间。\n\n"
        "没有修改runs、server_logs及正式汇总；simulations目录被Git忽略。"
        "完整生成规则、参考日志路径及SHA见SIMULATED_manifest.json。\n", encoding="utf-8")
    render_curves(output, all_runs, {"dark": args.dark_final, "foggy": args.foggy_final})
    print(json.dumps({"simulated": True, "output": str(output), "synthetic_runs": len(summaries), "epochs": args.epochs}))


if __name__ == "__main__":
    main()
