"""Audit a completed JMDA UDA log without using target labels for selection.

The audit separates protocol/configuration failures from performance warnings.
Low target accuracy never makes the protocol audit fail; it is reported only
as a diagnostic because UDA cannot use target labels to accept or reject a
checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


ITERATION_RE = re.compile(r"(?m)^迭代\s+(\d+)/(\d+)\s*$")
SUMMARY_RE = re.compile(r"(?m)^JMDA_RUN_SUMMARY_JSON:\s*(\{.*\})\s*$")
EPOCH_METRIC_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\].*?"
    r"src/tgt/mid=([-+0-9.eE]+)/([-+0-9.eE]+)/([-+0-9.eE]+).*?"
    r"labeled-target-observations=(\d+).*?"
    r"Source Train Acc:\s*([0-9.]+).*?Val Acc:\s*([0-9.]+)"
)
EPOCH_MACRO_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\].*?Val P/R/F1 \(Macro\):\s*"
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)"
)
EPOCH_SCHEDULE_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\].*?\| OT scale:\s*([-+0-9.eE]+).*?"
    r"\| LR feature/other/discriminator:\s*"
    r"([-+0-9.eE]+)/([-+0-9.eE]+)/([-+0-9.eE]+)"
)
SVD_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\]\s*\|\s*SVD update #(\d+):([^\r\n]*)"
)
WARMUP_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\]\s*\|\s*UDA source-only warm-up:"
)
SEED_RE = re.compile(r"(?m)^随机种子:\s*(-?\d+)\s*$")
MIN_COS_RE = re.compile(
    r"basis_min_cos_src/tgt=([-+0-9.eE]+)/([-+0-9.eE]+)"
)
CLASS_COUNT_RE = re.compile(r"classifier classes=(\d+)")
REPORT_RE = re.compile(
    r"Report strategy:\s*([a-z_]+),\s*window=(\d+),\s*"
    r"reported_acc=([-+0-9.eE]+)"
)
CODE_FINGERPRINT_RE = re.compile(r"训练代码指纹:\s*([0-9a-f]{16})")
PROJECTION_RE = re.compile(r"projection_dim=(\d+)")
ORACLE_MARKER = "TARGET LABEL SUPPORT ORACLE; NOT STRICT UDA."
DISPLAYED_METRIC_TOLERANCE = 5.0001e-5
FINGERPRINT_RELATIVE_PATHS = (
    "Ablation/module_ablation.py",
    "Tensor.py",
    "Generator.py",
    "Discriminator.py",
    "epoch_svd.py",
    "Ablation/Models.py",
    "Ablation/DataLoad.py",
    "Ablation/Tensor.py",
    "Ablation/Generator.py",
    "Ablation/Discriminator.py",
    "Ablation/epoch_svd.py",
    "Ablation/paired_dataset.py",
    "Ablation/PairedClassSampler.py",
)


def compute_training_code_fingerprint(jmda_net_dir: Path) -> str:
    """Mirror the training entry's portable source fingerprint calculation."""
    project_root = jmda_net_dir.resolve()
    digest = hashlib.sha256()
    for relative_path in FINGERPRINT_RELATIVE_PATHS:
        path = project_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Cannot fingerprint missing JMDA implementation file: {path}"
            )
        digest.update(relative_path.encode("utf-8"))
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(content)
    return digest.hexdigest()[:16]


def _segments(text: str) -> List[str]:
    starts = list(ITERATION_RE.finditer(text))
    return [
        text[match.start() : starts[index + 1].start()]
        if index + 1 < len(starts)
        else text[match.start() :]
        for index, match in enumerate(starts)
    ]


def _load_summary(text: str, errors: List[str]) -> Dict[str, Any]:
    matches = SUMMARY_RE.findall(text)
    if not matches:
        errors.append("missing JMDA_RUN_SUMMARY_JSON")
        return {}
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError as error:
        errors.append(f"invalid JMDA_RUN_SUMMARY_JSON: {error}")
        return {}


def _describe_integers(values: List[int]) -> str:
    """Keep audit errors readable for long runs without hiding their extent."""
    if len(values) <= 12:
        return repr(values)
    return (
        f"[{', '.join(map(str, values[:5]))}, ..., "
        f"{', '.join(map(str, values[-5:]))}] (count={len(values)})"
    )


def _parse_int_tuple_field(text: str, field: str) -> Optional[List[int]]:
    match = re.search(rf"{re.escape(field)}=\(([^)]*)\)", text)
    if match is None:
        return None
    values = [value.strip() for value in match.group(1).split(",")]
    try:
        return [int(value) for value in values if value]
    except ValueError:
        return None


def _parse_float_field(text: str, field: str) -> Optional[float]:
    match = re.search(
        rf"{re.escape(field)}=([-+0-9.eE]+)",
        text,
    )
    return float(match.group(1)) if match is not None else None


def _parse_int_field(text: str, field: str) -> Optional[int]:
    match = re.search(rf"{re.escape(field)}=(\d+)", text)
    return int(match.group(1)) if match is not None else None


def _parse_int_list_field(text: str, field: str) -> Optional[List[int]]:
    match = re.search(rf"{re.escape(field)}=\[([^]]*)\]", text)
    if match is None:
        return None
    values = [value.strip() for value in match.group(1).split(",")]
    try:
        return [int(value) for value in values if value]
    except ValueError:
        return None


def _parse_float_list_field(text: str, field: str) -> Optional[List[float]]:
    match = re.search(rf"{re.escape(field)}=\[([^]]*)\]", text)
    if match is None:
        return None
    values = [value.strip() for value in match.group(1).split(",")]
    try:
        return [float(value) for value in values if value]
    except ValueError:
        return None


def _expected_training_batches_without_singleton(
    sample_count: int,
    batch_size: int,
) -> int:
    """Mirror the sampler's singleton-tail merge for protocol auditing."""
    if sample_count < 1 or batch_size < 1:
        raise ValueError("sample_count and batch_size must be positive")
    batch_count = math.ceil(sample_count / batch_size)
    if batch_count > 1 and sample_count % batch_size == 1:
        batch_count -= 1
    return batch_count


def audit_text(
    text: str,
    *,
    expected_protocol: str,
    expected_projection_dim: int,
    expected_warmup_epochs: int,
    expected_total_epochs: int,
    expected_code_fingerprint: str,
    expected_num_iterations: int,
    expected_base_seed: int,
    expected_batch_size: Optional[int] = None,
    expected_svd_stat_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Return protocol errors, non-blocking warnings and per-seed diagnostics."""
    errors: List[str] = []
    warnings: List[str] = []
    details: List[Dict[str, Any]] = []

    fatal_markers = (
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "MemoryError",
        "FloatingPointError",
    )
    for marker in fatal_markers:
        if marker in text:
            errors.append(f"fatal marker present: {marker}")

    fingerprints = CODE_FINGERPRINT_RE.findall(text)
    if len(fingerprints) != 1:
        errors.append(
            "expected exactly one 16-hex training code fingerprint, "
            f"found {len(fingerprints)}"
        )
    fingerprint = fingerprints[0] if len(fingerprints) == 1 else None
    if fingerprint != expected_code_fingerprint:
        errors.append(
            "training code fingerprint mismatch: expected corrected source "
            f"{expected_code_fingerprint}, found {fingerprint}"
        )

    projection_values = {int(value) for value in PROJECTION_RE.findall(text)}
    if projection_values != {expected_projection_dim}:
        errors.append(
            f"projection_dim mismatch: expected {expected_projection_dim}, "
            f"found {sorted(projection_values)}"
        )

    summary = _load_summary(text, errors)
    pseudo_paired = (
        summary.get("uda_pairing_mode") == "source_prototype_pseudo_class"
    )
    summary_label_map = summary.get("label_map_original_to_index", {})
    summary_class_count = (
        len(summary_label_map) if isinstance(summary_label_map, dict) else 0
    )
    oracle_expected = expected_protocol == "ORACLE_SHARED_UDA"
    if oracle_expected and not pseudo_paired:
        errors.append(
            "ORACLE_SHARED_UDA must use source-prototype pseudo-class pairing"
        )
    if not oracle_expected and pseudo_paired:
        errors.append(
            "strict UDA must not use the oracle-only pseudo-class pairing mode"
        )
    if pseudo_paired and summary.get("target_ground_truth_used_for_pairing") is not False:
        errors.append(
            "pseudo-paired UDA summary does not prove target ground-truth "
            "labels were excluded from pair construction"
        )
    segments = _segments(text)
    if not segments:
        errors.append("no iteration sections found")

    iteration_headers = [
        (int(index), int(total))
        for index, total in ITERATION_RE.findall(text)
    ]
    expected_iteration_indices = list(range(1, len(iteration_headers) + 1))
    observed_iteration_indices = [index for index, _ in iteration_headers]
    if observed_iteration_indices != expected_iteration_indices:
        errors.append(
            "iteration header indices are not consecutive: expected "
            f"{expected_iteration_indices}, found {observed_iteration_indices}"
        )
    if any(total != len(iteration_headers) for _, total in iteration_headers):
        errors.append(
            "iteration header denominators do not equal the parsed iteration count"
        )
    if len(iteration_headers) != expected_num_iterations:
        errors.append(
            "iteration count mismatch: expected "
            f"{expected_num_iterations}, found {len(iteration_headers)}"
        )

    observed_seeds: List[int] = []
    final_accuracies: List[float] = []
    reported_accuracies: List[float] = []
    final_precision_macro: List[float] = []
    final_recall_macro: List[float] = []
    final_f1_macro: List[float] = []

    for iteration_index, segment in enumerate(segments, start=1):
        seed_matches = [int(seed) for seed in SEED_RE.findall(segment)]
        if len(seed_matches) != 1:
            errors.append(
                f"iteration {iteration_index}: expected exactly one random seed, "
                f"found {seed_matches}"
            )
        else:
            observed_seeds.append(seed_matches[0])

        label_exposure_marker = (
            "Target training item label exposure: include_label=False;"
        )
        if segment.count(label_exposure_marker) != 1:
            errors.append(
                f"iteration {iteration_index}: UDA target training items must "
                "prove include_label=False exactly once"
            )

        warmup_records = [
            (int(epoch), int(total))
            for epoch, total in WARMUP_RE.findall(segment)
        ]
        expected_warmup = list(range(1, expected_warmup_epochs + 1))
        observed_warmup = [epoch for epoch, _ in warmup_records]
        if observed_warmup != expected_warmup:
            errors.append(
                f"iteration {iteration_index}: source-only warm-up epochs "
                f"expected {_describe_integers(expected_warmup)}, "
                f"found {_describe_integers(observed_warmup)}"
            )
        if any(total != expected_total_epochs for _, total in warmup_records):
            errors.append(
                f"iteration {iteration_index}: warm-up total-epoch denominator mismatch"
            )

        svd_records = [
            (int(epoch), int(total), int(update), tail)
            for epoch, total, update, tail in SVD_RE.findall(segment)
        ]
        expected_svd_epochs = list(
            range(expected_warmup_epochs + 1, expected_total_epochs + 1)
        )
        observed_svd_epochs = [epoch for epoch, _, _, _ in svd_records]
        observed_updates = [update for _, _, update, _ in svd_records]
        if observed_svd_epochs != expected_svd_epochs:
            errors.append(
                f"iteration {iteration_index}: SVD epochs expected "
                f"{_describe_integers(expected_svd_epochs)}, "
                f"found {_describe_integers(observed_svd_epochs)}"
            )
        expected_updates = list(range(1, len(expected_svd_epochs) + 1))
        if observed_updates != expected_updates:
            errors.append(
                f"iteration {iteration_index}: SVD update numbers expected "
                f"{_describe_integers(expected_updates)}, "
                f"found {_describe_integers(observed_updates)}"
            )
        if any(total != expected_total_epochs for _, total, _, _ in svd_records):
            errors.append(
                f"iteration {iteration_index}: SVD total-epoch denominator mismatch"
            )
        if pseudo_paired:
            for svd_epoch, _, _, tail in svd_records:
                for required in (
                    "class_paired=True",
                    "class_balanced=True",
                    "target_labels_source=pseudo",
                ):
                    if required not in tail:
                        errors.append(
                            f"iteration {iteration_index}, epoch {svd_epoch}: "
                            f"pseudo-paired SVD is missing {required}"
                        )
            pseudo_pair_records = re.findall(
                r"Epoch \[(\d+)/(\d+)\]\s*\|\s*UDA pseudo-class pairing:"
                r"([^\r\n]*)",
                segment,
            )
            observed_pseudo_epochs = [
                int(epoch) for epoch, _, _ in pseudo_pair_records
            ]
            expected_pseudo_epochs = list(range(1, expected_total_epochs + 1))
            if observed_pseudo_epochs != expected_pseudo_epochs:
                errors.append(
                    f"iteration {iteration_index}: pseudo-pairing epochs expected "
                    f"{_describe_integers(expected_pseudo_epochs)}, found "
                    f"{_describe_integers(observed_pseudo_epochs)}"
                )
            for pseudo_epoch, _, tail in pseudo_pair_records:
                for required in (
                    "target_label_source=source-prototype (no target ground truth)",
                    "pseudo_feature_space=",
                    "class_counts=",
                    "coverage_repairs=",
                    "repaired-vs-nearest agreement=",
                    "modality_source_bal_acc=",
                    "modality_weights=",
                    "modality_reliability_estimator="
                    "leave-one-out-source-prototypes",
                    "change_rate=",
                    "target_samples=",
                    "balanced_epoch_samples=",
                    "training_batches=",
                ):
                    if required not in tail:
                        errors.append(
                            f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                            f"pseudo-pairing diagnostics are missing {required}"
                        )
                expected_pseudo_space = (
                    "native-pre-svd"
                    if int(pseudo_epoch) == 1
                    else "aligned-svd"
                )
                if (
                    f"pseudo_feature_space={expected_pseudo_space}"
                    not in tail
                ):
                    errors.append(
                        f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                        "pseudo feature space must be "
                        f"{expected_pseudo_space}"
                    )
                class_counts = _parse_int_list_field(tail, "class_counts")
                target_samples = _parse_int_field(tail, "target_samples")
                coverage_repairs = _parse_int_field(tail, "coverage_repairs")
                agreement = _parse_float_field(
                    tail,
                    "repaired-vs-nearest agreement",
                )
                balanced_epoch_samples = _parse_int_field(
                    tail,
                    "balanced_epoch_samples",
                )
                training_batches = _parse_int_field(tail, "training_batches")
                modality_accuracies = _parse_float_list_field(
                    tail,
                    "modality_source_bal_acc",
                )
                modality_weights = _parse_float_list_field(
                    tail,
                    "modality_weights",
                )
                if (
                    class_counts is None
                    or not class_counts
                    or any(count < 1 for count in class_counts)
                ):
                    errors.append(
                        f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                        f"invalid pseudo class_counts={class_counts!r}"
                    )
                else:
                    if (
                        summary_class_count
                        and len(class_counts) != summary_class_count
                    ):
                        errors.append(
                            f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                            "pseudo class count does not match classifier K: "
                            f"{len(class_counts)} != {summary_class_count}"
                        )
                    if target_samples is None or sum(class_counts) != target_samples:
                        errors.append(
                            f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                            "pseudo counts do not sum to target_samples: "
                            f"counts={class_counts}, target_samples={target_samples}"
                        )
                    expected_balanced_samples = max(class_counts) * len(class_counts)
                    if balanced_epoch_samples != expected_balanced_samples:
                        errors.append(
                            f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                            "balanced_epoch_samples mismatch: expected "
                            f"{expected_balanced_samples}, found "
                            f"{balanced_epoch_samples}"
                        )
                    if (
                        target_samples
                        and expected_balanced_samples / target_samples > 4.0
                    ):
                        warnings.append(
                            f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                            "pseudo-class balancing expands target exposure by "
                            f"{expected_balanced_samples / target_samples:.2f}x"
                        )
                if (
                    coverage_repairs is None
                    or coverage_repairs < 0
                    or (
                        summary_class_count
                        and coverage_repairs >= summary_class_count
                    )
                ):
                    errors.append(
                        f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                        f"invalid coverage_repairs={coverage_repairs!r}"
                    )
                if (
                    target_samples
                    and coverage_repairs is not None
                    and agreement is not None
                    and not math.isclose(
                        agreement,
                        1.0 - coverage_repairs / target_samples,
                        abs_tol=DISPLAYED_METRIC_TOLERANCE,
                    )
                ):
                    errors.append(
                        f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                        "repair agreement is inconsistent with coverage repairs"
                    )
                if (
                    modality_accuracies is None
                    or len(modality_accuracies) != 2
                    or any(
                        not math.isfinite(value) or not 0.0 <= value <= 1.0
                        for value in (modality_accuracies or [])
                    )
                ):
                    errors.append(
                        f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                        "invalid modality_source_bal_acc"
                    )
                if (
                    modality_weights is None
                    or len(modality_weights) != 2
                    or any(
                        not math.isfinite(value) or value <= 0.0
                        for value in (modality_weights or [])
                    )
                    or not math.isclose(
                        sum(modality_weights or []),
                        1.0,
                        abs_tol=1e-5,
                    )
                ):
                    errors.append(
                        f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                        "invalid modality_weights"
                    )
                summary_target_samples = summary.get("target_train_samples")
                if (
                    summary_target_samples is not None
                    and target_samples != summary_target_samples
                ):
                    errors.append(
                        f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                        "target_samples disagrees with run summary"
                    )
                summary_batch_size = (
                    summary.get("data_loading", {}).get("batch_size")
                    if isinstance(summary.get("data_loading"), dict)
                    else None
                )
                if balanced_epoch_samples and summary_batch_size:
                    expected_batches = _expected_training_batches_without_singleton(
                        balanced_epoch_samples,
                        summary_batch_size,
                    )
                    if training_batches != expected_batches:
                        errors.append(
                            f"iteration {iteration_index}, epoch {pseudo_epoch}: "
                            "training_batches mismatch: expected "
                            f"{expected_batches}, found {training_batches}"
                        )

        first_svd_tail = svd_records[0][3] if svd_records else ""
        first_transport_index = 0 if expected_warmup_epochs > 0 else 1
        first_transport_tail = (
            svd_records[first_transport_index][3]
            if len(svd_records) > first_transport_index
            else ""
        )
        min_cos_match = MIN_COS_RE.search(first_transport_tail)
        source_min_cosine = None
        target_min_cosine = None
        first_effective_ranks = _parse_int_tuple_field(
            first_svd_tail, "effective_ranks"
        )
        first_stabilized_null_dims = _parse_int_tuple_field(
            first_svd_tail, "stabilized_null_dims"
        )
        first_shared_transport_ranks = _parse_int_tuple_field(
            first_transport_tail, "shared_transport_ranks"
        )
        first_basis_domain_gap = _parse_float_field(
            first_transport_tail, "basis_domain_gap"
        )
        for field_name, values in (
            ("effective_ranks", first_effective_ranks),
            ("stabilized_null_dims", first_stabilized_null_dims),
        ):
            if values is None or len(values) != 2:
                errors.append(
                    f"iteration {iteration_index}: first SVD has invalid "
                    f"{field_name} tuple: {values!r}"
                )
        for field in ("effective_ranks=", "stabilized_null_dims=", "rank_thresholds="):
            if field not in first_svd_tail:
                errors.append(
                    f"iteration {iteration_index}: first SVD is missing {field}"
                )
        if (
            pseudo_paired
            and
            first_stabilized_null_dims is not None
            and first_stabilized_null_dims != [0, 0]
        ):
            errors.append(
                f"iteration {iteration_index}: legacy supervised paired-SVD must "
                "not stabilize null directions; found "
                f"{first_stabilized_null_dims}"
            )

        required_transition_fields = (
            "basis_nonorth_src/tgt=",
            "transport_nonorth=",
            "shared_transport_ranks=",
            "shared_transport_rank_thresholds=",
            "basis_domain_gap=",
            "classifier_basis=shared" if pseudo_paired else "classifier_basis=source",
            "basis_transport=legacy_supervised",
        )
        for field in required_transition_fields:
            if field not in first_transport_tail:
                errors.append(
                    f"iteration {iteration_index}: first transported SVD is "
                    f"missing {field}"
                )
        if min_cos_match is None:
            errors.append(
                f"iteration {iteration_index}: first transported SVD is missing "
                "basis_min_cos_src/tgt"
            )
        else:
            source_min_cosine = float(min_cos_match.group(1))
            target_min_cosine = float(min_cos_match.group(2))
        if first_shared_transport_ranks is None or len(first_shared_transport_ranks) != 2:
            errors.append(
                f"iteration {iteration_index}: first transported SVD has invalid "
                f"shared_transport_ranks tuple: {first_shared_transport_ranks!r}"
            )

        epoch_metric_matches = list(EPOCH_METRIC_RE.finditer(segment))
        epoch_metrics: Dict[int, Dict[str, float]] = {}
        epoch_metric_counts = Counter(
            int(match.group(1)) for match in epoch_metric_matches
        )
        duplicate_metric_epochs = sorted(
            epoch for epoch, count in epoch_metric_counts.items() if count != 1
        )
        if duplicate_metric_epochs:
            errors.append(
                f"iteration {iteration_index}: duplicate epoch metrics at "
                f"{_describe_integers(duplicate_metric_epochs)}"
            )
        for match in epoch_metric_matches:
            epoch = int(match.group(1))
            total = int(match.group(2))
            source_loss, target_loss, intermediate_loss = (
                float(match.group(3)),
                float(match.group(4)),
                float(match.group(5)),
            )
            labeled_target_observations = int(match.group(6))
            source_accuracy = float(match.group(7))
            validation_accuracy = float(match.group(8))
            if total != expected_total_epochs:
                errors.append(
                    f"iteration {iteration_index}, epoch {epoch}: total-epoch "
                    f"denominator is {total}, expected {expected_total_epochs}"
                )
            if abs(target_loss) > 1e-10:
                errors.append(
                    f"iteration {iteration_index}, epoch {epoch}: target "
                    "classification loss is non-zero"
                )
            if not pseudo_paired and abs(intermediate_loss) > 1e-10:
                errors.append(
                    f"iteration {iteration_index}, epoch {epoch}: unpaired UDA "
                    "intermediate classification loss is non-zero"
                )
            if labeled_target_observations != 0:
                errors.append(
                    f"iteration {iteration_index}, epoch {epoch}: observed "
                    f"{labeled_target_observations} labeled target samples"
                )
            epoch_metrics[epoch] = {
                "source_loss": source_loss,
                "source_accuracy": source_accuracy,
                "validation_accuracy": validation_accuracy,
            }

        expected_metric_epochs = set(range(1, expected_total_epochs + 1))
        if set(epoch_metrics) != expected_metric_epochs:
            missing = sorted(expected_metric_epochs - set(epoch_metrics))
            extra = sorted(set(epoch_metrics) - expected_metric_epochs)
            errors.append(
                f"iteration {iteration_index}: epoch metrics incomplete; "
                f"missing={missing}, extra={extra}"
            )

        schedule_records = {
            int(epoch): (
                int(total),
                float(ot_scale),
                float(feature_lr),
                float(other_lr),
                float(discriminator_lr),
            )
            for (
                epoch,
                total,
                ot_scale,
                feature_lr,
                other_lr,
                discriminator_lr,
            ) in EPOCH_SCHEDULE_RE.findall(segment)
        }
        if set(schedule_records) != expected_metric_epochs:
            errors.append(
                f"iteration {iteration_index}: OT/LR schedule records are incomplete"
            )
        ot_warmup_epochs = int(summary.get("ot_warmup_epochs", 10))
        ot_ramp_epochs = int(summary.get("ot_ramp_epochs", 5))
        expected_discriminator_lr = float(summary.get("lr_other", 2e-4))
        for schedule_epoch, (
            total,
            ot_scale,
            _feature_lr,
            _other_lr,
            discriminator_lr,
        ) in schedule_records.items():
            if total != expected_total_epochs:
                errors.append(
                    f"iteration {iteration_index}, epoch {schedule_epoch}: "
                    "OT/LR total-epoch denominator mismatch"
                )
            if schedule_epoch <= ot_warmup_epochs:
                expected_ot_scale = 0.0
            else:
                expected_ot_scale = min(
                    (schedule_epoch - ot_warmup_epochs) / ot_ramp_epochs,
                    1.0,
                )
            if not math.isclose(
                ot_scale,
                expected_ot_scale,
                abs_tol=5e-3,
            ):
                errors.append(
                    f"iteration {iteration_index}, epoch {schedule_epoch}: "
                    f"OT scale expected {expected_ot_scale:.4f}, "
                    f"found {ot_scale:.4f}"
                )
            if not math.isclose(
                discriminator_lr,
                expected_discriminator_lr,
                rel_tol=1e-6,
                abs_tol=1e-12,
            ):
                errors.append(
                    f"iteration {iteration_index}, epoch {schedule_epoch}: "
                    "discriminator LR is not fixed at the supervised-reference "
                    f"value {expected_discriminator_lr:g}: "
                    f"found {discriminator_lr:g}"
                )

        macro_matches = EPOCH_MACRO_RE.findall(segment)
        macro_counts = Counter(int(record[0]) for record in macro_matches)
        duplicate_macro_epochs = sorted(
            epoch for epoch, count in macro_counts.items() if count != 1
        )
        if duplicate_macro_epochs:
            errors.append(
                f"iteration {iteration_index}: duplicate macro epoch metrics at "
                f"{_describe_integers(duplicate_macro_epochs)}"
            )
        epoch_macro_metrics = {
            int(epoch): (
                float(precision),
                float(recall),
                float(f1),
            )
            for epoch, total, precision, recall, f1 in macro_matches
            if int(total) == expected_total_epochs
        }
        if set(epoch_macro_metrics) != expected_metric_epochs:
            missing = sorted(expected_metric_epochs - set(epoch_macro_metrics))
            extra = sorted(set(epoch_macro_metrics) - expected_metric_epochs)
            errors.append(
                f"iteration {iteration_index}: macro epoch metrics incomplete; "
                f"missing={_describe_integers(missing)}, "
                f"extra={_describe_integers(extra)}"
            )
        final_macro = epoch_macro_metrics.get(expected_total_epochs)
        if final_macro is not None:
            final_precision_macro.append(final_macro[0])
            final_recall_macro.append(final_macro[1])
            final_f1_macro.append(final_macro[2])

        final_accuracy = epoch_metrics.get(expected_total_epochs, {}).get(
            "validation_accuracy"
        )
        reports = REPORT_RE.findall(segment)
        if len(reports) != 1 or reports[0][0] != "last":
            errors.append(
                f"iteration {iteration_index}: expected one final-checkpoint "
                f"report, found reports={reports}"
            )
        elif final_accuracy is not None:
            reported_accuracy = float(reports[0][2])
            reported_accuracies.append(reported_accuracy)
            final_accuracies.append(final_accuracy)
            if abs(reported_accuracy - final_accuracy) > DISPLAYED_METRIC_TOLERANCE:
                errors.append(
                    f"iteration {iteration_index}: reported_acc "
                    f"{reported_accuracy:.8g} does not equal final-epoch Val Acc "
                    f"{final_accuracy:.8g}"
                )
        if "target-label best checkpoint is not selected or reported in UDA" not in segment:
            errors.append(
                f"iteration {iteration_index}: missing explicit no-target-best statement"
            )
        if "Saved final checkpoint:" not in segment:
            errors.append(f"iteration {iteration_index}: final checkpoint was not saved")

        class_matches = CLASS_COUNT_RE.findall(segment)
        class_count = (
            int(class_matches[-1])
            if class_matches
            else (summary_class_count or None)
        )
        if class_count is None or class_count < 1:
            errors.append(f"iteration {iteration_index}: classifier class count missing")

        if expected_protocol == "ORACLE_SHARED_UDA":
            if segment.count(ORACLE_MARKER) != 1:
                errors.append(
                    f"iteration {iteration_index}: expected exactly one oracle "
                    "target-label-support marker"
                )
        elif ORACLE_MARKER in segment:
            errors.append(
                f"iteration {iteration_index}: strict UDA contains an oracle "
                "target-label-support marker"
            )

        warmup_accuracy = (
            epoch_metrics.get(expected_warmup_epochs, {}).get("source_accuracy")
            if expected_warmup_epochs > 0
            else None
        )
        transition_epoch = expected_warmup_epochs + 1
        post_transition_accuracy = epoch_metrics.get(transition_epoch, {}).get(
            "source_accuracy"
        )
        chance_accuracy = 1.0 / class_count if class_count else None

        if (
            warmup_accuracy is not None
            and post_transition_accuracy is not None
            and warmup_accuracy - post_transition_accuracy > 0.20
        ):
            warnings.append(
                f"iteration {iteration_index}: source accuracy fell by more than "
                "0.20 across the first SVD transition"
            )
        if source_min_cosine is not None and source_min_cosine < 0.10:
            warnings.append(
                f"iteration {iteration_index}: first source basis minimum "
                f"principal cosine is {source_min_cosine:.3f} (<0.10)"
            )
        if (
            first_stabilized_null_dims is not None
            and any(null_dim > 0 for null_dim in first_stabilized_null_dims)
        ):
            warnings.append(
                f"iteration {iteration_index}: first post-warm-up projection "
                "contains stabilized numerical-null directions "
                f"{first_stabilized_null_dims}"
            )
        if (
            first_shared_transport_ranks is not None
            and any(
                rank < expected_projection_dim
                for rank in first_shared_transport_ranks
            )
        ):
            warnings.append(
                f"iteration {iteration_index}: first shared basis transport is "
                f"rank-deficient {first_shared_transport_ranks} relative to "
                f"projection_dim={expected_projection_dim}"
            )
        if (
            final_accuracy is not None
            and chance_accuracy is not None
            and final_accuracy < chance_accuracy
        ):
            warnings.append(
                f"iteration {iteration_index}: final target accuracy "
                f"{final_accuracy:.4f} is below 1/K chance {chance_accuracy:.4f}"
            )

        details.append(
            {
                "iteration": iteration_index,
                "class_count": class_count,
                "source_accuracy_before_transition": warmup_accuracy,
                "source_accuracy_after_transition_epoch": post_transition_accuracy,
                "source_min_principal_cosine": source_min_cosine,
                "target_min_principal_cosine": target_min_cosine,
                "first_effective_ranks": first_effective_ranks,
                "first_stabilized_null_dims": first_stabilized_null_dims,
                "first_shared_transport_ranks": first_shared_transport_ranks,
                "first_basis_domain_gap": first_basis_domain_gap,
                "final_target_accuracy": final_accuracy,
                "chance_accuracy": chance_accuracy,
            }
        )

    if summary:
        expected_label_space_mode = (
            "oracle_shared"
            if expected_protocol == "ORACLE_SHARED_UDA"
            else "partial_set"
        )
        expected_summary = {
            "protocol": expected_protocol,
            "adaptation_protocol": "UDA",
            "label_space_mode": expected_label_space_mode,
            "report_strategy": "last",
            "projection_dim": expected_projection_dim,
            "uda_source_warmup_epochs": expected_warmup_epochs,
        }
        for key, expected_value in expected_summary.items():
            if summary.get(key) != expected_value:
                errors.append(
                    f"summary {key!r} mismatch: expected {expected_value!r}, "
                    f"found {summary.get(key)!r}"
                )
        if summary.get("code_fingerprint") != fingerprint:
            errors.append("summary code_fingerprint does not match log header")
        if summary.get("num_iterations") != len(segments):
            errors.append(
                "summary num_iterations does not match parsed iteration sections"
            )
        summary_seeds = summary.get("seeds")
        if summary_seeds != observed_seeds:
            errors.append(
                f"summary seeds do not match iteration seeds: "
                f"summary={summary_seeds!r}, observed={observed_seeds!r}"
            )
        if len(observed_seeds) != len(set(observed_seeds)):
            errors.append(f"iteration seeds are not unique: {observed_seeds!r}")
        expected_seeds = list(
            range(expected_base_seed, expected_base_seed + expected_num_iterations)
        )
        if observed_seeds != expected_seeds:
            errors.append(
                f"iteration seeds mismatch: expected {expected_seeds!r}, "
                f"found {observed_seeds!r}"
            )
        if summary.get("oracle_target_label_support_used") is not oracle_expected:
            errors.append(
                "summary oracle_target_label_support_used is inconsistent with protocol"
            )
        oracle_labels = summary.get("oracle_shared_labels")
        if oracle_expected and not oracle_labels:
            errors.append("oracle summary is missing oracle_shared_labels")
        if not oracle_expected and oracle_labels not in ([], None):
            errors.append("strict UDA summary contains oracle_shared_labels")
        if pseudo_paired and not oracle_expected:
            errors.append(
                "source-prototype pseudo pairing is currently restricted to "
                "the separately reported oracle shared-label protocol"
            )
        expected_parity_fields = {
            "visual_finetuning_mode": "legacy_supervised",
            "adaptive_sinkhorn": False,
            "svd_projection_mode": "legacy_supervised",
            "basis_transport_mode": "legacy_supervised",
            "transport_mode": "sinkhorn",
            "ot_sinkhorn_iterations": 50,
            "ot_warmup_epochs": 10,
            "ot_ramp_epochs": 5,
            "class_conditional_ot": True,
            "balance_svd_classes": True,
            "sync_discriminator_lr": False,
            "tensor_loss_weight": 0.12,
            "adv_loss_weight": 0.08,
            "lr_feature": 5e-6,
            "lr_other": 2e-4,
            "weight_decay": 5e-4,
        }
        for key, expected_value in expected_parity_fields.items():
            if summary.get(key) != expected_value:
                errors.append(
                    f"summary {key!r} mismatch: expected {expected_value!r}, "
                    f"found {summary.get(key)!r}"
                )
        if summary.get("target_training_item_labels_exposed") is not False:
            errors.append(
                "summary does not prove UDA target training items excluded labels"
            )
        if pseudo_paired and summary.get("modality_reliability_estimator") != (
            "leave-one-out-source-prototypes"
        ):
            errors.append("summary modality reliability is not leave-one-out")
        data_loading = summary.get("data_loading")
        if not isinstance(data_loading, dict):
            errors.append("summary data_loading is missing")
        else:
            for key, expected_value in (
                ("batch_size", expected_batch_size),
                ("svd_stat_batch_size", expected_svd_stat_batch_size),
            ):
                if (
                    expected_value is not None
                    and data_loading.get(key) != expected_value
                ):
                    errors.append(
                        f"summary data_loading.{key} mismatch: expected "
                        f"{expected_value}, found {data_loading.get(key)!r}"
                    )

        summary_metrics = summary.get("metrics", {})
        expected_final_metrics = {
            "accuracy": final_accuracies,
            "precision_macro": final_precision_macro,
            "recall_macro": final_recall_macro,
            "f1_macro": final_f1_macro,
        }
        for metric_name, final_values in expected_final_metrics.items():
            metric_summary = summary_metrics.get(metric_name)
            if not isinstance(metric_summary, dict):
                errors.append(f"summary metrics.{metric_name} is missing")
                continue
            raw_values = metric_summary.get("values")
            try:
                summary_values = [float(value) for value in raw_values]
            except (TypeError, ValueError):
                summary_values = []
                errors.append(
                    f"summary {metric_name} values are not a numeric list"
                )
            if any(not math.isfinite(value) for value in summary_values):
                errors.append(
                    f"summary {metric_name} values contain non-finite numbers"
                )
            if len(summary_values) != len(final_values):
                errors.append(
                    f"summary {metric_name} values count does not match completed "
                    f"iterations: {len(summary_values)} != {len(final_values)}"
                )
            else:
                for index, (summary_value, final_value) in enumerate(
                    zip(summary_values, final_values), start=1
                ):
                    mismatch = (
                        abs(summary_value - final_value)
                        > DISPLAYED_METRIC_TOLERANCE
                    )
                    if metric_name == "accuracy" and index <= len(reported_accuracies):
                        mismatch = mismatch or (
                            abs(summary_value - reported_accuracies[index - 1])
                            > DISPLAYED_METRIC_TOLERANCE
                        )
                    if mismatch:
                        errors.append(
                            f"summary {metric_name} value {index} "
                            f"({summary_value:.8g}) "
                            "does not match that iteration's final/reported accuracy"
                        )
            if summary_values:
                expected_mean = sum(summary_values) / len(summary_values)
                expected_std = math.sqrt(
                    sum(
                        (value - expected_mean) ** 2
                        for value in summary_values
                    )
                    / len(summary_values)
                )
                for field, expected_value in (
                    ("mean", expected_mean),
                    ("std", expected_std),
                ):
                    try:
                        observed_value = float(metric_summary.get(field))
                    except (TypeError, ValueError):
                        errors.append(
                            f"summary {metric_name} {field} is missing or non-numeric"
                        )
                        continue
                    if (
                        not math.isfinite(observed_value)
                        or not math.isclose(
                            observed_value,
                            expected_value,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    ):
                        errors.append(
                            f"summary {metric_name} {field} mismatch: expected "
                            f"{expected_value:.12g}, found {observed_value:.12g}"
                        )

    return {
        "status": "FAIL" if errors else ("WARN" if warnings else "PASS"),
        "protocol_errors": errors,
        "performance_warnings": warnings,
        "code_fingerprint": fingerprint,
        "iterations": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a completed corrected JMDA UDA training log."
    )
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument(
        "--expected-protocol",
        required=True,
        choices=("UDA", "ORACLE_SHARED_UDA"),
    )
    parser.add_argument("--expected-projection-dim", required=True, type=int)
    parser.add_argument("--expected-warmup-epochs", required=True, type=int)
    parser.add_argument("--expected-total-epochs", required=True, type=int)
    parser.add_argument("--expected-code-fingerprint", required=True)
    parser.add_argument("--expected-num-iterations", required=True, type=int)
    parser.add_argument("--expected-base-seed", required=True, type=int)
    parser.add_argument("--expected-batch-size", required=True, type=int)
    parser.add_argument(
        "--expected-svd-stat-batch-size",
        required=True,
        type=int,
    )
    args = parser.parse_args()

    if args.expected_projection_dim < 1:
        parser.error("--expected-projection-dim must be positive")
    if args.expected_warmup_epochs < 0:
        parser.error("--expected-warmup-epochs cannot be negative")
    if args.expected_total_epochs <= args.expected_warmup_epochs:
        parser.error("--expected-total-epochs must exceed warm-up epochs")
    if not re.fullmatch(r"[0-9a-f]{16}", args.expected_code_fingerprint):
        parser.error("--expected-code-fingerprint must be 16 lowercase hex digits")
    if args.expected_num_iterations < 1:
        parser.error("--expected-num-iterations must be positive")
    if args.expected_batch_size < 1:
        parser.error("--expected-batch-size must be positive")
    if args.expected_svd_stat_batch_size < 1:
        parser.error("--expected-svd-stat-batch-size must be positive")
    if not args.log.is_file():
        parser.error(f"log file does not exist: {args.log}")

    result = audit_text(
        args.log.read_text(encoding="utf-8", errors="replace"),
        expected_protocol=args.expected_protocol,
        expected_projection_dim=args.expected_projection_dim,
        expected_warmup_epochs=args.expected_warmup_epochs,
        expected_total_epochs=args.expected_total_epochs,
        expected_code_fingerprint=args.expected_code_fingerprint,
        expected_num_iterations=args.expected_num_iterations,
        expected_base_seed=args.expected_base_seed,
        expected_batch_size=args.expected_batch_size,
        expected_svd_stat_batch_size=args.expected_svd_stat_batch_size,
    )
    print(f"JMDA log audit status: {result['status']}")
    for error in result["protocol_errors"]:
        print(f"  ERROR: {error}")
    for warning in result["performance_warnings"]:
        print(f"  WARN: {warning}")
    print("JMDA_LOG_AUDIT_JSON: " + json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 1 if result["protocol_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
