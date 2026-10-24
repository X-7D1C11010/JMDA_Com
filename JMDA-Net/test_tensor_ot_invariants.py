import unittest
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from Generator import NeuralOptimalTransportGenerator
from Tensor import TensorBasedAlignmentStable
from Ablation.Models import Classifier
from Ablation.PairedClassSampler import (
    PairedClassSampler,
    PseudoPairedClassSampler,
)
from Ablation.module_ablation import (
    BinaryDomainDiscriminator,
    UnpairedDomainSampler,
    _least_changing_orthogonal_factor,
    adaptation_protocol,
    prototype_pseudo_label_assignment,
    class_prior_to_sample_marginal,
    compute_protocol_classification_loss,
    compute_joint_ot_scale,
    configure_visual_finetuning,
    build_compact_label_map,
    derive_oracle_shared_labels,
    infer_prototype_pseudo_labels,
    is_uda_source_warmup,
    map_oracle_labels_to_pair_manifest,
    measure_final_model_efficiency,
    reported_protocol,
    select_report_metrics,
    set_frozen_batchnorm_eval,
    transport_projected_feature_basis,
    write_iteration_efficiency_artifacts,
)
from Ablation.paired_dataset import (
    LabelFreePairedTargetDataset,
    PairedModalTransform,
    PairedMultiModalDomainDataset,
    filter_samples_by_original_labels,
)
from epoch_svd import build_epoch_pairs, update_epoch_projections


class _LabelOnlyDataset:
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


class _FeatureDataset:
    """Small picklable dataset used to exercise SVD worker loading."""

    def __init__(self, count, offset=0.0):
        self.count = count
        self.offset = float(offset)
        self.labels = [index % 2 for index in range(count)]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        base = torch.tensor(
            [index + self.offset, index + 1.0, index * 0.5 + 2.0],
            dtype=torch.float32,
        )
        return {
            "vis": base,
            "ir": torch.flip(base, dims=(0,)),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


class _FeatureOnlyTargetDataset(_FeatureDataset):
    """Target feature bank with no accessible per-sample ground truth."""

    def __init__(self, count, offset=0.0):
        super().__init__(count, offset=offset)
        self.labels = None

    def __getitem__(self, index):
        base = torch.tensor(
            [index + self.offset, index + 1.0, index * 0.5 + 2.0],
            dtype=torch.float32,
        )
        return {
            "vis": base,
            "ir": torch.flip(base, dims=(0,)),
        }


class _TinyImageDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "vis": torch.full((3, 2, 2), float(index)),
            "ir": torch.full((3, 2, 2), float(index + 1)),
            "label": torch.tensor(index % 2, dtype=torch.long),
        }


class _TinyTal(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("vis_projection", torch.eye(4, 2))
        self.register_buffer("ir_projection", torch.eye(4, 2))

    @property
    def V_matrices(self):
        return [self.vis_projection, self.ir_projection]


class _PrototypeDataset:
    def __init__(self, vis, ir, labels):
        self.vis = [torch.tensor(row, dtype=torch.float32) for row in vis]
        self.ir = [torch.tensor(row, dtype=torch.float32) for row in ir]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "vis": self.vis[index],
            "ir": self.ir[index],
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


class _RandomizedLabelDataset(_LabelOnlyDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["worker_random"] = torch.tensor(
            [random.random(), float(np.random.random()), float(torch.rand(()))],
            dtype=torch.float64,
        )
        return sample


class EpochPairingTests(unittest.TestCase):
    def test_unpaired_sampler_covers_larger_domain_and_cycles_smaller(self):
        source = _LabelOnlyDataset([0, 1, 2, 3, 4])
        target = _LabelOnlyDataset([8, 9])
        sampler = UnpairedDomainSampler(source, target, batch_size=3)

        source_seen = []
        target_seen = []
        for source_batch, target_batch in sampler:
            source_seen.extend(source_batch['index'].tolist())
            target_seen.extend(target_batch['sample_index'].tolist())

        self.assertEqual(len(sampler), 2)
        self.assertEqual(set(source_seen), set(range(len(source))))
        self.assertEqual(set(target_seen), set(range(len(target))))

    def test_unpaired_sampler_worker_loader_covers_domains_deterministically(self):
        source = _RandomizedLabelDataset([0, 1, 2, 3, 4])
        target = _RandomizedLabelDataset([8, 9])

        def collect():
            sampler = UnpairedDomainSampler(
                source,
                target,
                batch_size=3,
                num_workers=1,
                prefetch_factor=1,
                pin_memory=False,
                seed=17,
            )
            source_seen = []
            target_seen = []
            worker_random = []
            for source_batch, target_batch in sampler:
                source_seen.extend(source_batch["index"].tolist())
                target_seen.extend(target_batch["sample_index"].tolist())
                worker_random.extend(source_batch["worker_random"].tolist())
                worker_random.extend(target_batch["worker_random"].tolist())
            return source_seen, target_seen, worker_random

        first_source, first_target, first_random = collect()
        second_source, second_target, second_random = collect()
        self.assertEqual(first_source, second_source)
        self.assertEqual(first_target, second_target)
        self.assertEqual(first_random, second_random)
        self.assertEqual(set(first_source), set(range(len(source))))
        self.assertEqual(set(first_target), set(range(len(target))))

    def test_unpaired_sampler_rejects_invalid_loader_parameters(self):
        source = _LabelOnlyDataset([0])
        target = _LabelOnlyDataset([1])
        with self.assertRaisesRegex(ValueError, "num_workers"):
            UnpairedDomainSampler(source, target, batch_size=1, num_workers=-1)
        with self.assertRaisesRegex(ValueError, "prefetch_factor"):
            UnpairedDomainSampler(
                source, target, batch_size=1, prefetch_factor=0
            )

    def test_epoch_svd_worker_loader_updates_complete_unpaired_banks(self):
        source = _FeatureDataset(5, offset=0.0)
        target = _FeatureDataset(3, offset=0.25)
        module = TensorBasedAlignmentStable(
            input_dims=[3, 3],
            output_dims=[2, 2],
            num_modalities=2,
            max_svd_sweeps=1,
        )

        update_info = update_epoch_projections(
            tal_module=module,
            encoders=[nn.Identity(), nn.Identity()],
            modality_keys=["vis", "ir"],
            source_dataset=source,
            target_dataset=target,
            batch_size=2,
            device=torch.device("cpu"),
            seed=23,
            class_paired=False,
            num_workers=1,
            prefetch_factor=1,
            pin_memory=False,
        )

        self.assertEqual(update_info["sample_count"], (5, 3))
        self.assertEqual(int(module.projection_update_count.item()), 1)

    def test_pseudo_paired_svd_never_requires_target_ground_truth(self):
        source = _FeatureDataset(6, offset=0.0)
        target = _FeatureOnlyTargetDataset(4, offset=0.25)
        module = TensorBasedAlignmentStable(
            input_dims=[3, 3],
            output_dims=[2, 2],
            num_modalities=2,
            max_svd_sweeps=1,
            stabilize_rank_deficient=False,
        )

        update_info = update_epoch_projections(
            tal_module=module,
            encoders=[nn.Identity(), nn.Identity()],
            modality_keys=["vis", "ir"],
            source_dataset=source,
            target_dataset=target,
            target_labels_override=[0, 1, 0, 1],
            batch_size=2,
            device=torch.device("cpu"),
            seed=29,
            class_paired=True,
            balance_classes=True,
            num_workers=0,
            prefetch_factor=1,
            pin_memory=False,
        )

        self.assertTrue(update_info["class_paired"])
        self.assertTrue(update_info["class_balanced"])
        self.assertEqual(update_info["target_labels_source"], "pseudo")
        self.assertEqual(int(module.projection_update_count.item()), 1)

    def test_class_prior_is_not_multiplied_by_source_class_frequency(self):
        labels = torch.tensor([0, 0, 0, 1])
        desired = torch.tensor([0.25, 0.75])

        marginal = class_prior_to_sample_marginal(labels, desired)

        self.assertAlmostEqual(float(marginal[labels == 0].sum()), 0.25, places=6)
        self.assertAlmostEqual(float(marginal[labels == 1].sum()), 0.75, places=6)

    def test_supervised_pairs_cover_target_once_and_match_classes(self):
        source = _LabelOnlyDataset([1, 1, 2, 2, 2])
        target = _LabelOnlyDataset([1, 2, 2, 1])
        pairs = build_epoch_pairs(source, target, seed=13, class_paired=True)

        target_indices = [target_index for _, target_index in pairs]
        self.assertEqual(sorted(target_indices), list(range(len(target))))
        self.assertEqual(len(set(target_indices)), len(target))
        self.assertTrue(
            all(
                source.labels[source_index] == target.labels[target_index]
                for source_index, target_index in pairs
            )
        )

    def test_supervised_training_sampler_covers_and_balances_target(self):
        source = _LabelOnlyDataset([1, 1, 1, 2, 2])
        target = _LabelOnlyDataset([1, 2, 2, 2])
        sampler = PairedClassSampler(source, target, batch_size=3)

        observed_target_indices = []
        observed_target_labels = []
        for source_batch, target_batch in sampler:
            self.assertTrue(
                torch.equal(source_batch["label"], target_batch["label"])
            )
            observed_target_indices.extend(target_batch["sample_index"].tolist())
            observed_target_labels.extend(target_batch["label"].tolist())

        self.assertEqual(set(observed_target_indices), set(range(len(target))))
        self.assertEqual(observed_target_labels.count(1), 3)
        self.assertEqual(observed_target_labels.count(2), 3)

    def test_pseudo_sampler_ignores_target_truth_and_matches_pseudo_classes(self):
        source = _LabelOnlyDataset([0, 0, 1, 1, 1])
        source.label_map = {0: 0, 1: 1}
        # Deliberately incompatible ground truth: the sampler must never use it.
        target = _LabelOnlyDataset([99, 99, 99, 99])
        pseudo = torch.tensor([0, 1, 1, 0])
        sampler = PseudoPairedClassSampler(
            source,
            target,
            pseudo,
            batch_size=3,
            expected_classes=[0, 1],
        )

        target_indices = []
        pseudo_observed = []
        for source_batch, target_batch in sampler:
            self.assertNotIn("label", target_batch)
            self.assertTrue(
                torch.equal(source_batch["label"], target_batch["pseudo_label"])
            )
            target_indices.extend(target_batch["sample_index"].tolist())
            pseudo_observed.extend(target_batch["pseudo_label"].tolist())

        self.assertEqual(set(target_indices), set(range(len(target))))
        self.assertEqual(pseudo_observed.count(0), pseudo_observed.count(1))

    def test_pseudo_sampler_merges_singleton_tail_without_dropping_samples(self):
        source = _LabelOnlyDataset(
            [class_index for class_index in range(5) for _ in range(3)]
        )
        source.label_map = {class_index: class_index for class_index in range(5)}
        pseudo = torch.tensor(
            [0] * 13 + [1] * 10 + [2] * 9 + [3] * 8 + [4] * 7
        )
        target = _LabelOnlyDataset([99] * pseudo.numel())
        sampler = PseudoPairedClassSampler(
            source,
            target,
            pseudo,
            batch_size=32,
            expected_classes=range(5),
        )

        batch_sizes = []
        observed_target_indices = set()
        for source_batch, target_batch in sampler:
            batch_sizes.append(int(source_batch["label"].numel()))
            observed_target_indices.update(target_batch["sample_index"].tolist())
            self.assertTrue(
                torch.equal(source_batch["label"], target_batch["pseudo_label"])
            )

        # Every target sample appears once. The natural pseudo distribution is
        # not expanded to 5 * max_count(13) = 65 observations.
        self.assertEqual(batch_sizes, [32, 15])
        self.assertEqual(len(sampler), 2)
        self.assertEqual(sampler.epoch_sample_count, len(target))
        self.assertEqual(observed_target_indices, set(range(len(target))))

    def test_pseudo_svd_pairing_uses_override_not_target_truth(self):
        source = _LabelOnlyDataset([10, 10, 20, 20])
        target = _LabelOnlyDataset([99, 99, 99, 99])
        pseudo_original = [10, 20, 20, 10]
        pairs = build_epoch_pairs(
            source,
            target,
            seed=29,
            class_paired=True,
            balance_classes=True,
            target_labels_override=pseudo_original,
        )
        self.assertTrue(
            all(
                source.labels[source_index] == pseudo_original[target_index]
                for source_index, target_index in pairs
            )
        )

    def test_prototype_assignment_preserves_nearest_distribution_without_repairs(self):
        scores = torch.tensor(
            [
                [9.0, 1.0, 0.0],
                [8.0, 2.0, 0.0],
                [7.0, 3.0, 0.0],
                [6.0, 4.0, 0.0],
                [5.0, 5.0, 0.0],
                [4.0, 6.0, 0.0],
                [3.0, 7.0, 0.0],
            ]
        )
        nearest = scores.argmax(dim=1)
        labels, info = prototype_pseudo_label_assignment(scores)
        self.assertEqual(labels.numel(), 7)
        self.assertEqual(set(labels.tolist()), {0, 1})
        self.assertEqual(info["repaired_count"], 0)
        self.assertTrue(torch.equal(labels, nearest))
        self.assertTrue(
            torch.equal(
                torch.bincount(labels, minlength=3),
                info["class_counts"],
            )
        )
        self.assertEqual(info["class_counts"].tolist(), [5, 2, 0])

    def test_prototype_inference_keeps_natural_counts_and_downweights_weak_modality(self):
        source = _PrototypeDataset(
            vis=[[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]],
            ir=[[1, 1], [1, 1], [1, 1], [1, 1]],
            labels=[0, 0, 1, 1],
        )
        target = _PrototypeDataset(
            vis=[[1, 0], [0.9, 0.1], [0.8, 0.2], [1, 0], [0, 1]],
            ir=[[1, 1]] * 5,
            # Deliberately invalid target truth; inference must not consume it.
            labels=[99] * 5,
        )

        labels, info = infer_prototype_pseudo_labels(
            net_vis=nn.Identity(),
            net_ir=nn.Identity(),
            tal_module=nn.Identity(),
            source_dataset=source,
            target_dataset=target,
            num_classes=2,
            batch_size=3,
            device=torch.device("cpu"),
            use_tensor_module=False,
        )

        self.assertEqual(labels.tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(info["class_counts"], [4, 1])
        self.assertEqual(info["coverage_repairs"], 0)
        self.assertGreater(info["modality_weights"][0], 0.99)
        self.assertLess(info["modality_weights"][1], 0.01)
        self.assertEqual(
            info["modality_reliability_estimator"],
            "leave-one-out-source-prototypes",
        )

    def test_random_features_do_not_get_in_sample_prototype_reliability(self):
        torch.manual_seed(20261021)
        num_classes = 9
        samples_per_class = 29
        source_features = torch.randn(
            num_classes * samples_per_class,
            512,
        )
        target_features = torch.randn(18, 512)
        source = _PrototypeDataset(
            vis=source_features.tolist(),
            ir=source_features.tolist(),
            labels=torch.arange(num_classes).repeat_interleave(
                samples_per_class
            ).tolist(),
        )
        target = _PrototypeDataset(
            vis=target_features.tolist(),
            ir=target_features.tolist(),
            labels=[99] * len(target_features),
        )

        _, info = infer_prototype_pseudo_labels(
            net_vis=nn.Identity(),
            net_ir=nn.Identity(),
            tal_module=nn.Identity(),
            source_dataset=source,
            target_dataset=target,
            num_classes=num_classes,
            batch_size=64,
            device=torch.device("cpu"),
            use_tensor_module=False,
        )

        # With the old self-containing prototypes this synthetic random bank
        # scored near 1.0.  Leave-one-out must stay close to chance (1/9).
        self.assertLess(max(info["modality_balanced_accuracy"]), 0.30)

    def test_paired_transform_uses_identical_random_geometry(self):
        grid = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
        image = Image.fromarray(grid, mode="L").convert("RGB")
        transform = PairedModalTransform(phase="train")
        torch.manual_seed(123)

        vis, ir = transform(image, image)
        vis_raw = vis * torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        vis_raw = vis_raw + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        ir_raw = ir * 0.5 + 0.5

        self.assertTrue(torch.allclose(vis_raw, ir_raw, atol=1e-6))

    def test_uda_target_dataset_redacts_internal_and_returned_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vis_dir = root / "train" / "1" / "可见光"
            ir_dir = root / "train" / "1" / "红外"
            vis_dir.mkdir(parents=True)
            ir_dir.mkdir(parents=True)
            image = Image.new("RGB", (32, 32), color=(64, 96, 128))
            image.save(vis_dir / "sample.png")
            image.save(ir_dir / "sample.png")

            dataset = PairedMultiModalDomainDataset(
                str(root),
                domain_type="target",
                phase="train",
                global_label_map={1: 0},
                include_label=False,
            )

            self.assertIsNone(dataset.labels)
            self.assertNotIn("label", dataset.samples[0])
            self.assertNotIn("label", dataset[0])

    def test_oracle_target_training_loader_consumes_only_label_free_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vis_path = root / "sample.png"
            ir_path = root / "sample_ir.png"
            # Exact-pair manifests require the same stem, so place modalities
            # in separate folders as in the real dataset.
            vis_path = root / "可见光" / "sample.png"
            ir_path = root / "红外" / "sample.png"
            vis_path.parent.mkdir(parents=True)
            ir_path.parent.mkdir(parents=True)
            image = Image.new("RGB", (32, 32), color=(64, 96, 128))
            image.save(vis_path)
            image.save(ir_path)
            pair_list = root / "target_pairs.tsv"
            pair_list.write_text(
                f"{vis_path}\t{ir_path}\n",
                encoding="utf-8",
            )

            dataset = LabelFreePairedTargetDataset(
                pair_list,
                num_classes=3,
            )
            self.assertIsNone(dataset.labels)
            self.assertEqual(dataset.num_classes, 3)
            self.assertNotIn("label", dataset.samples[0])
            self.assertNotIn("label", dataset[0])

    def test_oracle_pair_labels_are_joined_by_exact_paths_not_row_order(self):
        label_free = SimpleNamespace(
            samples=[
                {"vis": "target/b/vis.png", "ir": "target/b/ir.png"},
                {"vis": "target/a/vis.png", "ir": "target/a/ir.png"},
            ]
        )
        oracle = SimpleNamespace(
            samples=[
                {"vis": "target/a/vis.png", "ir": "target/a/ir.png"},
                {"vis": "target/b/vis.png", "ir": "target/b/ir.png"},
            ],
            labels=[10, 20],
        )

        mapped = map_oracle_labels_to_pair_manifest(
            label_free,
            oracle,
            {10: 0, 20: 1},
        )

        self.assertTrue(torch.equal(mapped, torch.tensor([1, 0])))


class ResultReportingTests(unittest.TestCase):
    def test_joint_ot_warmup_and_ramp_schedule(self):
        scales = [
            compute_joint_ot_scale(
                epoch, warmup_epochs=3, ramp_epochs=2, enabled=True
            )
            for epoch in range(6)
        ]
        self.assertEqual(scales, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0])
        self.assertEqual(
            compute_joint_ot_scale(0, 3, 2, enabled=False),
            1.0,
        )

    def test_uda_source_warmup_is_protocol_scoped(self):
        self.assertTrue(is_uda_source_warmup("UDA", epoch=0, warmup_epochs=10))
        self.assertTrue(is_uda_source_warmup("UDA", epoch=9, warmup_epochs=10))
        self.assertFalse(is_uda_source_warmup("UDA", epoch=10, warmup_epochs=10))
        self.assertFalse(is_uda_source_warmup("SDA", epoch=0, warmup_epochs=10))
        self.assertFalse(is_uda_source_warmup("SSDA", epoch=0, warmup_epochs=10))
        with self.assertRaises(ValueError):
            is_uda_source_warmup("UDA", epoch=0, warmup_epochs=-1)

    def test_best_strategy_keeps_metrics_from_one_checkpoint(self):
        history = [
            {
                "accuracy": 0.4,
                "precision_macro": 0.9,
                "recall_macro": 0.8,
                "f1_macro": 0.7,
            },
            {
                "accuracy": 0.8,
                "precision_macro": 0.3,
                "recall_macro": 0.4,
                "f1_macro": 0.35,
            },
            {
                "accuracy": 0.6,
                "precision_macro": 0.7,
                "recall_macro": 0.6,
                "f1_macro": 0.65,
            },
        ]

        selected = select_report_metrics(history, strategy="best")

        self.assertIs(selected, history[1])
        self.assertEqual(selected["accuracy"], 0.8)
        self.assertEqual(selected["precision_macro"], 0.3)


class AdaptationProtocolTests(unittest.TestCase):
    @staticmethod
    def _args(use_target_labels=False, ratio=0.0, weight=0.0,
              label_space_mode="partial_set",
              diagnostic_oracle_pairing=False,
              diagnostic_source_only=False,
              label_guided_alignment=False):
        return SimpleNamespace(
            use_target_labels=use_target_labels,
            target_label_ratio=ratio,
            target_cls_weight=weight,
            label_space_mode=label_space_mode,
            diagnostic_oracle_pairing=diagnostic_oracle_pairing,
            diagnostic_source_only=diagnostic_source_only,
            label_guided_alignment=label_guided_alignment,
        )

    def test_protocol_states_are_disjoint(self):
        self.assertEqual(adaptation_protocol(self._args(True)), "SDA")
        self.assertEqual(adaptation_protocol(self._args(False, 0.2, 0.5)), "SSDA")
        self.assertEqual(adaptation_protocol(self._args()), "UDA")

    def test_oracle_shared_has_separate_reported_protocol(self):
        args = self._args(label_space_mode="oracle_shared")
        self.assertEqual(adaptation_protocol(args), "UDA")
        self.assertEqual(reported_protocol(args), "ORACLE_SHARED_UDA")
        with self.assertRaisesRegex(ValueError, "only defined for pure UDA"):
            reported_protocol(self._args(True, label_space_mode="oracle_shared"))

    def test_oracle_pairing_and_source_only_diagnostics_have_unique_protocols(self):
        oracle_pairing = self._args(
            label_space_mode="oracle_shared",
            diagnostic_oracle_pairing=True,
        )
        source_only = self._args(
            label_space_mode="oracle_shared",
            diagnostic_source_only=True,
        )
        self.assertEqual(
            reported_protocol(oracle_pairing),
            "ORACLE_PAIRED_UDA_DIAGNOSTIC",
        )
        self.assertEqual(
            reported_protocol(source_only),
            "ORACLE_SHARED_SOURCE_ONLY_DIAGNOSTIC",
        )

    def test_label_guided_alignment_has_formal_reported_protocol(self):
        args = self._args(
            label_space_mode="oracle_shared",
            label_guided_alignment=True,
        )
        self.assertEqual(adaptation_protocol(args), "UDA")
        self.assertEqual(
            reported_protocol(args),
            "LABEL_GUIDED_DA_NO_TARGET_CE",
        )

    def test_oracle_shared_label_intersection_is_explicit(self):
        self.assertEqual(
            derive_oracle_shared_labels([1, 2, 3, 3], [3, 4, 1]),
            [1, 3],
        )
        with self.assertRaisesRegex(ValueError, "no shared classes"):
            derive_oracle_shared_labels([1], [2])

    def test_oracle_shared_label_map_is_compact_and_deterministic(self):
        self.assertEqual(
            build_compact_label_map([14, 1, 9, 5, 9]),
            {1: 0, 5: 1, 9: 2, 14: 3},
        )
        with self.assertRaisesRegex(ValueError, "empty class support"):
            build_compact_label_map([])

    def test_oracle_filter_keeps_original_ids_and_default_is_unchanged(self):
        samples = [
            {"label": 1, "name": "a"},
            {"label": 4, "name": "b"},
            {"label": 9, "name": "c"},
        ]
        self.assertEqual(filter_samples_by_original_labels(samples), samples)
        filtered = filter_samples_by_original_labels(samples, {1, 9})
        self.assertEqual([sample["label"] for sample in filtered], [1, 9])
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            filter_samples_by_original_labels(samples, set())


class ProtocolClassificationLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.criterion = nn.CrossEntropyLoss()
        self.source_labels = torch.tensor([0, 1, 2])
        self.target_labels = torch.tensor([2, 1, 0])

    def _logits(self):
        return torch.randn(3, 3, requires_grad=True)

    def test_sda_contains_source_full_target_and_class_paired_mid_ce(self):
        source = self._logits()
        target = self._logits()
        intermediate = self._logits()
        scale = 0.4

        total, terms = compute_protocol_classification_loss(
            self.criterion,
            "SDA",
            source,
            self.source_labels,
            target_logits=target,
            target_labels=self.target_labels,
            target_cls_weight=1.0,
            intermediate_logits=intermediate,
            intermediate_scale=scale,
        )

        expected = (
            self.criterion(source, self.source_labels)
            + self.criterion(target, self.target_labels)
            + scale * self.criterion(intermediate, self.source_labels)
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertEqual(terms["target_labeled_count"], 3)
        total.backward()
        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(target.grad)
        self.assertIsNotNone(intermediate.grad)

    def test_ssda_uses_only_fixed_target_subset_and_ignores_mid_logits(self):
        source = self._logits()
        target = self._logits()
        intermediate = self._logits()
        target_indices = torch.tensor([10, 11, 12])

        total, terms = compute_protocol_classification_loss(
            self.criterion,
            "SSDA",
            source,
            self.source_labels,
            target_logits=target,
            target_labels=self.target_labels,
            target_label_ratio=0.5,
            target_cls_weight=0.7,
            target_indices=target_indices,
            target_labeled_indices={11},
            intermediate_logits=intermediate,
        )

        expected = (
            self.criterion(source, self.source_labels)
            + 0.7 * self.criterion(target[1:2], self.target_labels[1:2])
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertEqual(terms["target_labeled_count"], 1)
        self.assertEqual(float(terms["intermediate"]), 0.0)
        total.backward()
        self.assertIsNone(intermediate.grad)

    def test_uda_requires_no_target_labels_and_has_source_ce_only(self):
        source = self._logits()
        target = self._logits()
        intermediate = self._logits()

        total, terms = compute_protocol_classification_loss(
            self.criterion,
            "UDA",
            source,
            self.source_labels,
            target_logits=target,
            target_labels=None,
            target_cls_weight=1.0,
            intermediate_logits=intermediate,
        )

        self.assertTrue(
            torch.allclose(total, self.criterion(source, self.source_labels))
        )
        self.assertEqual(terms["target_labeled_count"], 0)
        total.backward()
        self.assertIsNone(target.grad)
        self.assertIsNone(intermediate.grad)

    def test_labeled_protocols_require_target_labels(self):
        for protocol in ("SDA", "SSDA"):
            with self.subTest(protocol=protocol):
                with self.assertRaisesRegex(ValueError, "requires target logits"):
                    compute_protocol_classification_loss(
                        self.criterion,
                        protocol,
                        self._logits(),
                        self.source_labels,
                    )

    def test_invalid_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported adaptation protocol"):
            compute_protocol_classification_loss(
                self.criterion,
                "DG",
                self._logits(),
                self.source_labels,
            )


class VisualFineTuningTests(unittest.TestCase):
    class _DummyVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                    for _ in range(8)
                ]
            )
            self.proj = nn.Linear(2, 2)

    def test_resnet_stage_indices_and_frozen_bn_modes(self):
        model = self._DummyVisual()
        configure_visual_finetuning(model, mode="layer2_4")

        for stage_index in range(5):
            self.assertFalse(
                any(parameter.requires_grad for parameter in model.features[stage_index].parameters())
            )
        for stage_index in (5, 6, 7):
            self.assertTrue(
                all(parameter.requires_grad for parameter in model.features[stage_index].parameters())
            )
        self.assertTrue(all(parameter.requires_grad for parameter in model.proj.parameters()))

        model.train()
        frozen_count = set_frozen_batchnorm_eval(model)
        self.assertEqual(frozen_count, 5)
        for stage_index in range(5):
            self.assertFalse(model.features[stage_index][1].training)
        for stage_index in (5, 6, 7):
            self.assertTrue(model.features[stage_index][1].training)

    def test_legacy_supervised_mode_trains_only_projection_and_keeps_bn_train(self):
        model = self._DummyVisual()
        configure_visual_finetuning(model, mode="legacy_supervised")
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.features.parameters()
            )
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.proj.parameters())
        )
        model.train()
        self.assertTrue(
            all(
                child[1].training
                for child in model.features
            )
        )


class TensorEpochUpdateTests(unittest.TestCase):
    def test_unpaired_transport_supports_different_domain_sizes(self):
        torch.manual_seed(19)
        module = TensorBasedAlignmentStable(
            input_dims=[6, 5],
            output_dims=[3, 3],
            num_modalities=2,
            max_svd_sweeps=2,
            unpaired_sinkhorn_iterations=100,
        )
        source = [torch.randn(9, 6), torch.randn(9, 5)]
        target = [torch.randn(7, 6), torch.randn(7, 5)]
        source_weights = torch.tensor(
            [0.05, 0.05, 0.10, 0.10, 0.10, 0.15, 0.15, 0.15, 0.15]
        )

        projected_source = [
            features.matmul(projection)
            for features, projection in zip(source, module.U_matrices)
        ]
        projected_target = [
            features.matmul(projection)
            for features, projection in zip(target, module.V_matrices)
        ]
        plan, details = module.build_unlabeled_transport_plan(
            projected_source, projected_target, source_weights=source_weights
        )

        self.assertEqual(tuple(plan.shape), (9, 7))
        self.assertTrue(
            torch.allclose(plan.sum(1), details['source_marginal'], atol=2e-4)
        )
        self.assertTrue(
            torch.allclose(plan.sum(0), details['target_marginal'], atol=2e-4)
        )
        info = module.update_unpaired_projections(
            source, target, source_weights=source_weights
        )
        self.assertEqual(info['sample_count'], (9, 7))
        self.assertEqual(info['update_count'], 1)

    def test_downstream_basis_transport_preserves_pure_rotation(self):
        torch.manual_seed(3)
        modal_rank = 3
        feature_dim = 2 * modal_rank
        old_source = [
            torch.linalg.qr(torch.randn(7, modal_rank)).Q,
            torch.linalg.qr(torch.randn(8, modal_rank)).Q,
        ]
        old_target = [matrix.clone() for matrix in old_source]
        rotations = [
            torch.linalg.qr(torch.randn(modal_rank, modal_rank)).Q
            for _ in range(2)
        ]
        new_source = [
            projection.matmul(rotation)
            for projection, rotation in zip(old_source, rotations)
        ]
        new_target = [matrix.clone() for matrix in new_source]
        basis_map = torch.block_diag(*rotations)

        classifier = Classifier(input_dim=feature_dim, num_classes=4).eval()
        discriminator = BinaryDomainDiscriminator(feature_dim=feature_dim).eval()
        generator = NeuralOptimalTransportGenerator(
            feature_dim=feature_dim,
            hidden_dim=10,
            transport_mode="sinkhorn",
        ).eval()
        old_features = torch.randn(5, feature_dim)
        new_features = old_features.matmul(basis_map)

        classifier_expected = classifier(old_features)
        discriminator_expected = discriminator(old_features)
        cost_expected = generator.cost_net(old_features, old_features)
        residual_input = torch.cat(
            [old_features, old_features, torch.full((5, 1), 0.4)], dim=1
        )
        residual_expected = generator.mlp[:3](residual_input).matmul(basis_map)

        diagnostics = transport_projected_feature_basis(
            classifier,
            discriminator,
            generator,
            old_source,
            old_target,
            new_source,
            new_target,
        )

        self.assertTrue(
            torch.allclose(classifier(new_features), classifier_expected, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(
                discriminator(new_features), discriminator_expected, atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                generator.cost_net(new_features, new_features),
                cost_expected,
                atol=1e-5,
            )
        )
        new_residual_input = torch.cat(
            [new_features, new_features, torch.full((5, 1), 0.4)], dim=1
        )
        self.assertTrue(
            torch.allclose(
                generator.mlp[:3](new_residual_input), residual_expected, atol=1e-5
            )
        )
        self.assertAlmostEqual(
            diagnostics["source_subspace_overlap"], 1.0, places=5
        )
        self.assertAlmostEqual(
            diagnostics["target_subspace_overlap"], 1.0, places=5
        )
        self.assertAlmostEqual(
            diagnostics["source_min_principal_cosine"], 1.0, places=5
        )
        self.assertAlmostEqual(
            diagnostics["target_min_principal_cosine"], 1.0, places=5
        )
        self.assertLess(diagnostics["basis_map_non_orthogonality"], 1e-5)
        self.assertLess(
            diagnostics["source_basis_map_non_orthogonality"], 1e-5
        )
        self.assertLess(
            diagnostics["target_basis_map_non_orthogonality"], 1e-5
        )

    def test_rank_deficient_shared_polar_keeps_unidentified_axes_unchanged(self):
        matrix = torch.diag(torch.tensor([-1.0, 0.0, 0.0]))
        transport, effective_rank, threshold = (
            _least_changing_orthogonal_factor(matrix)
        )

        self.assertEqual(effective_rank, 1)
        self.assertGreater(threshold, 0.0)
        self.assertTrue(
            torch.allclose(
                transport,
                torch.diag(torch.tensor([-1.0, 1.0, 1.0])),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                transport.transpose(0, 1).matmul(transport),
                torch.eye(3),
                atol=1e-6,
            )
        )

    def test_legacy_supervised_basis_transport_uses_raw_average_everywhere(self):
        feature_dim = 2
        old_source = [torch.eye(4)[:, :2]]
        old_target = [torch.eye(4)[:, :2]]
        new_source = [torch.eye(4)[:, [0, 2]]]
        new_target = [torch.eye(4)[:, [2, 1]]]
        expected_map = 0.5 * torch.eye(feature_dim)

        classifier = Classifier(input_dim=feature_dim, num_classes=3)
        discriminator = BinaryDomainDiscriminator(feature_dim=feature_dim)
        generator = NeuralOptimalTransportGenerator(
            feature_dim=feature_dim,
            hidden_dim=5,
            transport_mode="sinkhorn",
        )
        originals = {
            "classifier": classifier.fc[0].weight.detach().clone(),
            "discriminator": discriminator.discriminator[0].weight.detach().clone(),
            "cost": generator.cost_net.proj[0].weight.detach().clone(),
            "residual_input": generator.mlp[0].weight.detach().clone(),
            "residual_output": generator.mlp[2].weight.detach().clone(),
            "residual_bias": generator.mlp[2].bias.detach().clone(),
        }

        diagnostics = transport_projected_feature_basis(
            classifier,
            discriminator,
            generator,
            old_source,
            old_target,
            new_source,
            new_target,
            classifier_basis_mode="shared",
            shared_transport_mode="legacy_supervised",
        )

        self.assertTrue(torch.allclose(
            classifier.fc[0].weight,
            originals["classifier"].matmul(expected_map),
        ))
        self.assertTrue(torch.allclose(
            discriminator.discriminator[0].weight,
            originals["discriminator"].matmul(expected_map),
        ))
        self.assertTrue(torch.allclose(
            generator.cost_net.proj[0].weight,
            originals["cost"].matmul(expected_map),
        ))
        expected_residual_input = originals["residual_input"].clone()
        expected_residual_input[:, :feature_dim] = (
            expected_residual_input[:, :feature_dim].matmul(expected_map)
        )
        expected_residual_input[:, feature_dim:2 * feature_dim] = (
            expected_residual_input[:, feature_dim:2 * feature_dim].matmul(
                expected_map
            )
        )
        self.assertTrue(torch.allclose(
            generator.mlp[0].weight,
            expected_residual_input,
        ))
        self.assertTrue(torch.allclose(
            generator.mlp[2].weight,
            expected_map.transpose(0, 1).matmul(originals["residual_output"]),
        ))
        self.assertTrue(torch.allclose(
            generator.mlp[2].bias,
            expected_map.transpose(0, 1).matmul(originals["residual_bias"]),
        ))
        self.assertEqual(
            diagnostics["basis_transport_mode"],
            "legacy_supervised",
        )
        self.assertGreater(
            diagnostics["shared_transport_non_orthogonality"],
            0.1,
        )

    def test_uda_classifier_transport_is_anchored_to_source_basis(self):
        torch.manual_seed(37)
        modal_rank = 3
        feature_dim = 2 * modal_rank
        old_source = [
            torch.linalg.qr(torch.randn(7, modal_rank)).Q,
            torch.linalg.qr(torch.randn(8, modal_rank)).Q,
        ]
        old_target = [
            torch.linalg.qr(torch.randn(7, modal_rank)).Q,
            torch.linalg.qr(torch.randn(8, modal_rank)).Q,
        ]
        source_rotations = [
            torch.linalg.qr(torch.randn(modal_rank, modal_rank)).Q
            for _ in range(2)
        ]
        target_rotations = [
            torch.linalg.qr(torch.randn(modal_rank, modal_rank)).Q
            for _ in range(2)
        ]
        new_source = [
            projection.matmul(rotation)
            for projection, rotation in zip(old_source, source_rotations)
        ]
        new_target = [
            projection.matmul(rotation)
            for projection, rotation in zip(old_target, target_rotations)
        ]
        source_map = torch.block_diag(*source_rotations)

        classifier = Classifier(input_dim=feature_dim, num_classes=4).eval()
        discriminator = BinaryDomainDiscriminator(feature_dim=feature_dim).eval()
        generator = NeuralOptimalTransportGenerator(
            feature_dim=feature_dim,
            hidden_dim=10,
            transport_mode="sinkhorn",
        ).eval()
        old_source_features = torch.randn(5, feature_dim)
        new_source_features = old_source_features.matmul(source_map)
        expected = classifier(old_source_features)
        shared_parameter_norms = {
            "discriminator": discriminator.discriminator[0].weight.norm().clone(),
            "cost": generator.cost_net.proj[0].weight.norm().clone(),
            "residual_input": generator.mlp[0].weight.norm().clone(),
            "residual_output": generator.mlp[2].weight.norm().clone(),
        }

        diagnostics = transport_projected_feature_basis(
            classifier,
            discriminator,
            generator,
            old_source,
            old_target,
            new_source,
            new_target,
            classifier_basis_mode="source",
        )

        self.assertTrue(
            torch.allclose(classifier(new_source_features), expected, atol=1e-5)
        )
        self.assertEqual(diagnostics["classifier_basis_mode"], "source")
        self.assertGreater(diagnostics["source_target_basis_map_gap"], 1e-3)
        self.assertGreater(diagnostics["basis_map_non_orthogonality"], 1e-3)
        self.assertLess(
            diagnostics["source_basis_map_non_orthogonality"], 1e-5
        )
        self.assertLess(
            diagnostics["target_basis_map_non_orthogonality"], 1e-5
        )
        self.assertLess(
            diagnostics["shared_transport_non_orthogonality"], 1e-5
        )
        self.assertTrue(
            torch.allclose(
                discriminator.discriminator[0].weight.norm(),
                shared_parameter_norms["discriminator"],
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                generator.cost_net.proj[0].weight.norm(),
                shared_parameter_norms["cost"],
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                generator.mlp[0].weight.norm(),
                shared_parameter_norms["residual_input"],
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                generator.mlp[2].weight.norm(),
                shared_parameter_norms["residual_output"],
                atol=1e-5,
            )
        )

    def test_basis_transport_resets_only_reparameterized_adam_state(self):
        torch.manual_seed(43)
        modal_rank = 2
        feature_dim = 2 * modal_rank
        old_source = [
            torch.linalg.qr(torch.randn(6, modal_rank)).Q,
            torch.linalg.qr(torch.randn(7, modal_rank)).Q,
        ]
        old_target = [projection.clone() for projection in old_source]
        rotations = [
            torch.linalg.qr(torch.randn(modal_rank, modal_rank)).Q
            for _ in range(2)
        ]
        new_source = [
            projection.matmul(rotation)
            for projection, rotation in zip(old_source, rotations)
        ]
        new_target = [projection.clone() for projection in new_source]

        classifier = Classifier(input_dim=feature_dim, num_classes=3)
        discriminator = BinaryDomainDiscriminator(feature_dim=feature_dim)
        generator = NeuralOptimalTransportGenerator(
            feature_dim=feature_dim,
            hidden_dim=8,
            transport_mode="sinkhorn",
        )
        optimizer_g = torch.optim.Adam(
            list(classifier.parameters()) + list(generator.parameters()),
            lr=1e-3,
        )
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

        optimizer_g.zero_grad()
        sum(
            parameter.square().mean()
            for parameter in list(classifier.parameters())
            + list(generator.parameters())
        ).backward()
        optimizer_g.step()
        optimizer_d.zero_grad()
        sum(
            parameter.square().mean()
            for parameter in discriminator.parameters()
        ).backward()
        optimizer_d.step()

        transformed_g = (
            classifier.fc[0].weight,
            generator.cost_net.proj[0].weight,
            generator.mlp[0].weight,
            generator.mlp[2].weight,
            generator.mlp[2].bias,
        )
        transformed_d = (discriminator.discriminator[0].weight,)
        untouched_g = classifier.fc[0].bias
        untouched_d = discriminator.discriminator[0].bias
        self.assertTrue(all(parameter in optimizer_g.state for parameter in transformed_g))
        self.assertTrue(all(parameter in optimizer_d.state for parameter in transformed_d))
        self.assertIn(untouched_g, optimizer_g.state)
        self.assertIn(untouched_d, optimizer_d.state)

        transport_projected_feature_basis(
            classifier,
            discriminator,
            generator,
            old_source,
            old_target,
            new_source,
            new_target,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            classifier_basis_mode="source",
        )

        self.assertTrue(all(parameter not in optimizer_g.state for parameter in transformed_g))
        self.assertTrue(all(parameter not in optimizer_d.state for parameter in transformed_d))
        self.assertIn(untouched_g, optimizer_g.state)
        self.assertIn(untouched_d, optimizer_d.state)

    def test_paired_procrustes_alignment_removes_subspace_rotation(self):
        torch.manual_seed(5)
        new_source = torch.linalg.qr(torch.randn(9, 4)).Q
        new_target = torch.linalg.qr(torch.randn(8, 4)).Q
        rotation = torch.linalg.qr(torch.randn(4, 4)).Q
        old_source = new_source.matmul(rotation)
        old_target = new_target.matmul(rotation)

        aligned_source, aligned_target = (
            TensorBasedAlignmentStable._align_paired_subspace(
                new_source,
                new_target,
                old_source,
                old_target,
            )
        )

        self.assertTrue(torch.allclose(aligned_source, old_source, atol=1e-5))
        self.assertTrue(torch.allclose(aligned_target, old_target, atol=1e-5))

    def test_rank_deficient_completion_keeps_identified_prefix_orthogonal(self):
        torch.manual_seed(41)
        old_basis = torch.linalg.qr(torch.randn(12, 6)).Q
        informative_basis = torch.linalg.qr(torch.randn(12, 2)).Q

        completed = (
            TensorBasedAlignmentStable._least_changing_orthogonal_completion(
                informative_basis,
                old_basis,
                total_rank=6,
            )
        )

        self.assertEqual(tuple(completed.shape), (12, 6))
        self.assertTrue(
            torch.allclose(
                completed.transpose(0, 1).matmul(completed),
                torch.eye(6),
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(completed[:, :2], informative_basis, atol=1e-5)
        )

    def test_effective_rank_uses_float32_machine_precision(self):
        covariance = torch.zeros(512, 512, dtype=torch.float32)
        singular_values = torch.tensor([1.0, 3.0e-5], dtype=torch.float32)

        effective_rank, threshold = TensorBasedAlignmentStable._effective_rank(
            covariance,
            singular_values,
        )

        self.assertEqual(effective_rank, 1)
        self.assertGreater(threshold, float(singular_values[1]))

    def test_rank_deficient_pair_retains_previous_null_subspaces(self):
        torch.manual_seed(42)
        old_source = torch.linalg.qr(torch.randn(12, 6)).Q
        old_target = torch.linalg.qr(torch.randn(11, 6)).Q
        new_source = torch.linalg.qr(torch.randn(12, 6)).Q
        new_target = torch.linalg.qr(torch.randn(11, 6)).Q
        new_source[:, :2] = old_source[:, :2]
        new_target[:, :2] = old_target[:, :2]

        stable_source, stable_target = (
            TensorBasedAlignmentStable._stabilize_rank_deficient_pair(
                new_source,
                new_target,
                old_source,
                old_target,
                informative_rank=2,
            )
        )

        self.assertTrue(
            torch.allclose(
                stable_source.matmul(stable_source.transpose(0, 1)),
                old_source.matmul(old_source.transpose(0, 1)),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                stable_target.matmul(stable_target.transpose(0, 1)),
                old_target.matmul(old_target.transpose(0, 1)),
                atol=1e-5,
            )
        )

    def test_rank_deficient_update_reports_stabilized_null_dimensions(self):
        torch.manual_seed(43)
        module = TensorBasedAlignmentStable(
            input_dims=[8, 6],
            output_dims=[4, 4],
            num_modalities=2,
            max_svd_sweeps=1,
        )
        latent_source = torch.randn(12, 1)
        latent_target = torch.randn(10, 1)
        source = [
            latent_source.matmul(torch.randn(1, 8)),
            latent_source.matmul(torch.randn(1, 6)),
        ]
        target = [
            latent_target.matmul(torch.randn(1, 8)),
            latent_target.matmul(torch.randn(1, 6)),
        ]

        info = module.update_unpaired_projections(source, target)

        self.assertIn("stabilized_null_dims", info)
        self.assertIn("rank_thresholds", info)
        self.assertTrue(any(value > 0 for value in info["stabilized_null_dims"]))
        for projection in (*module.U_matrices, *module.V_matrices):
            self.assertTrue(
                torch.allclose(
                    projection.transpose(0, 1).matmul(projection),
                    torch.eye(4),
                    atol=1e-5,
                    rtol=1e-5,
                )
            )

    def test_legacy_supervised_paired_svd_does_not_replace_null_directions(self):
        torch.manual_seed(44)
        module = TensorBasedAlignmentStable(
            input_dims=[8, 6],
            output_dims=[4, 4],
            num_modalities=2,
            max_svd_sweeps=1,
            stabilize_rank_deficient=False,
        )
        latent = torch.randn(12, 1)
        source = [
            latent.matmul(torch.randn(1, 8)),
            latent.matmul(torch.randn(1, 6)),
        ]
        target = [
            latent.matmul(torch.randn(1, 8)),
            latent.matmul(torch.randn(1, 6)),
        ]

        info = module.update_projections(source, target)

        self.assertEqual(info["stabilized_null_dims"], (0, 0))
        self.assertTrue(any(rank < 4 for rank in info["effective_ranks"]))
        for projection in (*module.U_matrices, *module.V_matrices):
            self.assertTrue(
                torch.allclose(
                    projection.transpose(0, 1).matmul(projection),
                    torch.eye(4),
                    atol=1e-5,
                    rtol=1e-5,
                )
            )

    def test_epoch_update_is_orthogonal_and_forward_is_read_only(self):
        torch.manual_seed(7)
        module = TensorBasedAlignmentStable(
            input_dims=[8, 6],
            output_dims=[4, 4],
            num_modalities=2,
            max_svd_sweeps=3,
        )
        source_bank = [torch.randn(16, 8), torch.randn(16, 6)]
        target_bank = [
            source_bank[0] + 0.05 * torch.randn(16, 8),
            source_bank[1] + 0.05 * torch.randn(16, 6),
        ]

        info = module.update_projections(source_bank, target_bank)
        self.assertEqual(info["update_count"], 1)
        self.assertEqual(info["sample_count"], 16)
        for projection in (*module.U_matrices, *module.V_matrices):
            identity = torch.eye(projection.shape[1])
            self.assertTrue(
                torch.allclose(
                    projection.transpose(0, 1).matmul(projection),
                    identity,
                    atol=1e-5,
                    rtol=1e-5,
                )
            )

        before = [matrix.clone() for matrix in (*module.U_matrices, *module.V_matrices)]
        source_batch = [torch.randn(5, 8, requires_grad=True), torch.randn(5, 6, requires_grad=True)]
        target_batch = [torch.randn(5, 8, requires_grad=True), torch.randn(5, 6, requires_grad=True)]
        _, _, loss = module(source_batch, target_batch)
        loss.backward()

        self.assertTrue(all(feature.grad is not None for feature in source_batch + target_batch))
        after = [matrix for matrix in (*module.U_matrices, *module.V_matrices)]
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, after)))
        self.assertEqual(int(module.projection_update_count.item()), 1)

    def test_per_batch_update_is_rejected(self):
        module = TensorBasedAlignmentStable(
            input_dims=[8, 6], output_dims=[4, 4], num_modalities=2
        )
        source = [torch.randn(5, 8), torch.randn(5, 6)]
        target = [torch.randn(5, 8), torch.randn(5, 6)]
        with self.assertRaisesRegex(RuntimeError, "Per-mini-batch"):
            module(source, target, update_projections=True)

    def test_rank_guard_rejects_unsupported_projection(self):
        module = TensorBasedAlignmentStable(
            input_dims=[8, 6], output_dims=[4, 4], num_modalities=2
        )
        source = [torch.randn(4, 8), torch.randn(4, 6)]
        target = [torch.randn(4, 8), torch.randn(4, 6)]
        with self.assertRaisesRegex(ValueError, "maximum centered rank=3"):
            module.update_projections(source, target)


class SinkhornTransportTests(unittest.TestCase):
    def test_sinkhorn_is_fixed_iteration_by_default(self):
        generator = NeuralOptimalTransportGenerator(
            feature_dim=6,
            hidden_dim=8,
            transport_mode="sinkhorn",
            sinkhorn_iterations=50,
        )
        source = torch.randn(5, 6)
        target = torch.randn(5, 6)
        _, details = generator(source, target, return_details=True)
        self.assertEqual(int(details["sinkhorn_iterations_used"]), 50)
        self.assertEqual(generator.sinkhorn_max_iterations, 50)

    def test_sinkhorn_accepts_unlabeled_nonuniform_source_marginal(self):
        torch.manual_seed(23)
        generator = NeuralOptimalTransportGenerator(
            feature_dim=8,
            hidden_dim=10,
            transport_mode='sinkhorn',
            sinkhorn_iterations=100,
            adaptive_sinkhorn=True,
        )
        source = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(3, 8, requires_grad=True)
        source_marginal = torch.tensor([0.1, 0.2, 0.3, 0.4])

        intermediate, details = generator(
            source,
            target,
            return_details=True,
            source_marginal=source_marginal,
        )

        self.assertTrue(
            torch.allclose(
                details['mass_plan'].sum(1), source_marginal, atol=2e-4
            )
        )
        self.assertTrue(
            torch.allclose(
                details['mass_plan'].sum(0), torch.full((3,), 1 / 3), atol=2e-4
            )
        )
        self.assertGreater(int(details['sinkhorn_iterations_used'].item()), 100)
        self.assertLess(float(details['marginal_residual'].detach()), 2e-4)
        intermediate.square().mean().backward()
        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(target.grad)

    def test_legacy_mode_keeps_original_transnet_row_softmax(self):
        torch.manual_seed(9)
        generator = NeuralOptimalTransportGenerator(
            feature_dim=10,
            hidden_dim=12,
            transport_mode="legacy_row_softmax",
        )
        generator.eval()
        source = torch.randn(6, 10)
        target = torch.randn(6, 10)

        cost = generator.cost_net(source, target)
        expected = torch.softmax(generator.transmission_net(cost), dim=1)
        _, details = generator(source, target, return_details=True)
        self.assertTrue(
            torch.allclose(details["conditional_plan"], expected, atol=1e-7)
        )

    def test_sinkhorn_enforces_both_marginals_and_backpropagates(self):
        torch.manual_seed(11)
        generator = NeuralOptimalTransportGenerator(
            feature_dim=12,
            hidden_dim=16,
            transport_mode="sinkhorn",
            epsilon=0.1,
            sinkhorn_iterations=100,
            correction_scale=0.1,
        )
        source = torch.randn(7, 12, requires_grad=True)
        target = torch.randn(5, 12, requires_grad=True)
        intermediate, details = generator(source, target, return_details=True)

        plan = details["mass_plan"]
        expected_rows = details["source_marginal"]
        expected_columns = details["target_marginal"]
        self.assertTrue(
            torch.allclose(
                details["softmax_kernel"].sum(dim=1),
                torch.ones(7),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.allclose(plan.sum(dim=1), expected_rows, atol=2e-4))
        self.assertTrue(torch.allclose(plan.sum(dim=0), expected_columns, atol=2e-4))
        self.assertLess(float(details["marginal_residual"]), 2e-4)
        self.assertTrue(
            torch.allclose(
                details["conditional_plan"].sum(dim=1),
                torch.ones(7),
                atol=2e-4,
            )
        )

        loss = intermediate.square().mean() + details[
            "correction_regularization"
        ]
        loss.backward()
        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(target.grad)
        transnet_gradients = [
            parameter.grad
            for parameter in generator.transmission_net.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in transnet_gradients))

    def test_supervised_sinkhorn_has_zero_off_class_mass(self):
        torch.manual_seed(17)
        generator = NeuralOptimalTransportGenerator(
            feature_dim=6,
            hidden_dim=8,
            transport_mode="sinkhorn",
            epsilon=0.1,
            sinkhorn_iterations=100,
        )
        source = torch.randn(4, 6)
        target = torch.randn(4, 6)
        labels = torch.tensor([0, 0, 1, 1])

        _, details = generator(
            source,
            target,
            return_details=True,
            source_labels=labels,
            target_labels=labels,
        )

        plan = details["mass_plan"]
        off_class = labels[:, None].ne(labels[None, :])
        self.assertEqual(float(plan.masked_select(off_class).sum()), 0.0)
        self.assertLess(float(details["marginal_residual"]), 2e-4)


class EfficiencyEvidenceTests(unittest.TestCase):
    def test_exact_inference_graph_records_flops_latency_and_artifacts(self):
        device = torch.device("cpu")
        net_vis = nn.Sequential(nn.Flatten(), nn.Linear(12, 4), nn.ReLU())
        net_ir = nn.Sequential(nn.Flatten(), nn.Linear(12, 4), nn.ReLU())
        tal = _TinyTal()
        classifier = nn.Linear(4, 2)
        generator = nn.Linear(4, 4)
        discriminator = nn.Linear(4, 3)
        loader = torch.utils.data.DataLoader(_TinyImageDataset(), batch_size=2)

        metrics = measure_final_model_efficiency(
            net_vis=net_vis,
            net_ir=net_ir,
            tal_module=tal,
            generator=generator,
            classifier=classifier,
            discriminator=discriminator,
            val_loader=loader,
            device=device,
            use_tensor_module=True,
            warmup_iterations=1,
            benchmark_iterations=3,
            require_flops=True,
        )
        self.assertGreater(metrics["training_total_params"], 0)
        self.assertGreater(metrics["training_trainable_params"], 0)
        self.assertGreater(metrics["inference_gflops"], 0)
        self.assertGreater(metrics["inference_latency_mean_ms"], 0)
        self.assertGreater(metrics["inference_throughput_samples_s"], 0)
        self.assertFalse(metrics["inference_data_loading_included"])

        with tempfile.TemporaryDirectory() as directory:
            json_path, csv_path = write_iteration_efficiency_artifacts(
                output_dir=directory,
                target_weather="逆光",
                seed=42,
                protocol="LABEL_GUIDED_DA_NO_TARGET_CE",
                epoch_records=[{
                    "epoch": 1,
                    "epoch_time_s": 1.0,
                    "svd_update_time_s": 0.2,
                    "peak_allocated_gb": 0.0,
                    "peak_reserved_gb": 0.0,
                }],
                metrics=metrics,
            )
            self.assertTrue(Path(json_path).is_file())
            self.assertTrue(Path(csv_path).is_file())


if __name__ == "__main__":
    unittest.main()
