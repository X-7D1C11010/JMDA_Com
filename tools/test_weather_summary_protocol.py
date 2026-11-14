import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from weather.summarize_vds_core import summarize


def record(root, folder, backbone="resnet18", frozen=False, normalization="none", method="dann", layout="reference", fda=0., zoom=1.):
    directory = root/folder
    directory.mkdir()
    metrics = {"accuracy": .8, "precision_macro": .8, "recall_macro": .8, "f1_macro": .8, "val_samples": 36}
    complexity = {k: 1. for k in ("total_params", "trainable_params", "inference_GFLOPs", "total_train_time_s",
                                "train_epoch_mean_s", "latency_mean_ms", "throughput_samples_s", "peak_train_reserved_gb")}
    result = {"core": "vds2raw.adaptation.AdaptationModel", "target_training_labels_used": False,
              "method": method, "target": "backlight", "seed": 42, "selected_epoch": 1,
              "completed_epochs": 2, "optimizer_updates": 4, "target_val": metrics,
              "evaluation_protocol": "best_target_val_within_budget", "smoke": False,
              "complexity": complexity, "image_backbone": backbone, "freeze_image_encoders": frozen,
              "feature_normalization": normalization, "classifier_layout": layout,
              "input_augmentation": {"FDA_beta": fda, "source_zoom_min_area": zoom}}
    (directory/"result.json").write_text(json.dumps(result))
    (directory/"completeness.json").write_text('{"status":"PASS"}')


class WeatherSummaryProtocolTests(unittest.TestCase):
    def test_rejects_mixed_encoder_freeze_and_fusion_protocols(self):
        for changed in ({"backbone": "convnext_small"}, {"frozen": True}, {"normalization": "unit"},
                        {"layout": "linear"}, {"fda": .01}, {"zoom": .25}):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                record(root, "dann_seed42")
                record(root, "mcc_seed42", method="mcc", **changed)
                with self.assertRaisesRegex(ValueError, "encoder or fusion-normalization"):
                    summarize(root)
                self.assertFalse((root/"summary").exists())

    def test_keeps_protocol_columns_and_rejects_duplicate_seed_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record(root, "dann_seed42", backbone="convnext_small", frozen=True, normalization="unit")
            with contextlib.redirect_stdout(io.StringIO()):
                rows = summarize(root)
            self.assertEqual((rows[0]["image_backbone"], rows[0]["freeze_image_encoders"], rows[0]["feature_normalization"]),
                             ("convnext_small", True, "unit"))
            record(root, "dann_trial_seed42", backbone="convnext_small", frozen=True, normalization="unit")
            with self.assertRaisesRegex(ValueError, "Repeated method/target/seed"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
