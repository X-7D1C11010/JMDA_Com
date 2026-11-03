"""Diagnostics for the shared peak-within-budget reporting protocol."""


def diagnose_peak_timing(history, total_budget, threshold=0.70, tolerance=1e-12):
    """Classify whether the maximum is reached or recovered late in a run."""
    if not history:
        raise ValueError("peak timing history must not be empty")
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    normalized = [(int(step), float(accuracy)) for step, accuracy in history]
    peak_accuracy = max(accuracy for _, accuracy in normalized)
    peak_steps = [
        step for step, accuracy in normalized
        if abs(accuracy - peak_accuracy) <= tolerance
    ]
    last_peak_step = max(peak_steps)
    last_peak_fraction = last_peak_step / float(total_budget)
    return {
        "peak_accuracy": peak_accuracy,
        "peak_steps": peak_steps,
        "last_peak_step": last_peak_step,
        "last_peak_fraction": last_peak_fraction,
        "threshold": threshold,
        "status": (
            "late_peak_or_recovered"
            if last_peak_fraction >= threshold
            else "early_peak_not_recovered"
        ),
    }


def format_peak_timing(diagnostic, unit="epoch"):
    peak_steps = ",".join(str(step) for step in diagnostic["peak_steps"])
    return (
        "Peak timing diagnostic: "
        f"peak_acc={diagnostic['peak_accuracy']:.4f}; "
        f"matching_{unit}s=[{peak_steps}]; "
        f"last_peak_progress={diagnostic['last_peak_fraction']:.1%}; "
        f"threshold={diagnostic['threshold']:.1%}; "
        f"status={diagnostic['status']}"
    )


__all__ = ["diagnose_peak_timing", "format_peak_timing"]
