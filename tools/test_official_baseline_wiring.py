import argparse
import importlib.util
import os
import sys
import tempfile
import unittest

import torch
from PIL import Image


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(name, path, prepend=None):
    if prepend and prepend not in sys.path:
        sys.path.insert(0, prepend)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialBaselineWiringTests(unittest.TestCase):
    def test_dann_uses_released_cnn_with_dynamic_head(self):
        model_dir = os.path.join(ROOT, "DANN", "models")
        module = load_module(
            "jmda_dann_official_model",
            os.path.join(model_dir, "model.py"),
            prepend=model_dir,
        )
        model = module.CNNModel(num_classes=7).eval()
        class_out, domain_out = model(torch.randn(3, 3, 28, 28), 0.5)
        self.assertEqual(tuple(class_out.shape), (3, 7))
        self.assertEqual(tuple(domain_out.shape), (3, 2))

    def test_deepcoral_uses_released_alexnet(self):
        path = os.path.join(ROOT, "DeepCORAL", "models.py")
        module = load_module("jmda_deepcoral_official_model", path)
        model = module.DeepCORAL(num_classes=5).eval()
        source, target = model(
            torch.randn(2, 3, 227, 227),
            torch.randn(2, 3, 227, 227),
        )
        self.assertEqual(tuple(source.shape), (2, 5))
        self.assertEqual(tuple(target.shape), (2, 5))

    def test_entrypoints_declare_official_defaults(self):
        with open(os.path.join(ROOT, "DANN", "train_dann.py"), encoding="utf-8") as f:
            dann = f.read()
        with open(os.path.join(ROOT, "DeepCORAL", "train_deepcoral.py"), encoding="utf-8") as f:
            coral = f.read()
        with open(os.path.join(ROOT, "CDAN", "train_cdan.py"), encoding="utf-8") as f:
            cdan = f.read()
        self.assertIn("default=128", dann)
        self.assertIn("default=200", coral)
        self.assertIn("default=56", coral)
        self.assertIn("choices=['ResNet18', 'ResNet50']", cdan)
        self.assertIn("default='ResNet18'", cdan)
        self.assertIn("resnet_name=args.backbone", cdan)

    def test_dann_can_decouple_grl_ramp_from_shortened_budget(self):
        with open(os.path.join(ROOT, "DANN", "train_dann.py"), encoding="utf-8") as f:
            dann = f.read()
        with open(os.path.join(ROOT, "scripts", "run_uda_sunny.ps1"),
                  encoding="utf-8-sig") as f:
            launcher = f.read()
        self.assertIn("--grl-max-iterations", dann)
        self.assertIn("grl_max_iterations = args.grl_max_iterations or planned_updates", dann)
        self.assertIn("[int]$DANNGrlMaxIterations = 0", launcher)
        self.assertIn("'--grl-max-iterations', $DANNGrlMaxIterations", launcher)

    def test_pda_peak_tie_uses_tolerant_accuracy_comparison(self):
        with open(os.path.join(ROOT, "PDA", "official", "trainers", "baseda.py"),
                  encoding="utf-8") as f:
            trainer = f.read()
        self.assertIn("acc_tolerance = 1e-9", trainer)
        self.assertIn("tied_acc_better_f1", trainer)

    def test_evaluation_loaders_do_not_add_worker_pools_by_default(self):
        tools_dir = os.path.join(ROOT, "tools")
        module = load_module(
            "jmda_data_loader_options",
            os.path.join(tools_dir, "data_loader_options.py"),
            prepend=tools_dir,
        )
        parser = argparse.ArgumentParser()
        module.add_data_loader_args(parser, default_workers=4)
        args = parser.parse_args([])
        self.assertEqual(module.data_loader_kwargs(args)["num_workers"], 4)
        self.assertEqual(module.data_loader_kwargs(args)["timeout"], 300)
        self.assertEqual(module.evaluation_loader_kwargs(args)["num_workers"], 0)

    def test_cdan_ten_crop_does_not_start_ten_loader_pools(self):
        with open(os.path.join(ROOT, "CDAN", "train_cdan.py"), encoding="utf-8") as f:
            cdan = f.read()
        self.assertIn("for transform in transform_list", cdan)
        self.assertNotIn("iterators = [iter(loader) for loader in loaders]", cdan)

    def test_local_launcher_has_windows_worker_guard_and_unbuffered_logs(self):
        with open(os.path.join(ROOT, "scripts", "run_uda_sunny.ps1"),
                  encoding="utf-8-sig") as f:
            launcher = f.read()
        self.assertIn("WindowsWorkerCap = 2", launcher)
        self.assertIn("$env:PYTHONUNBUFFERED = '1'", launcher)
        self.assertIn("@('-B', '-u', $scriptPath", launcher)
        self.assertIn("[string[]]$IncludeMethods", launcher)

    def test_uda_launchers_split_methods_by_default_and_opt_in(self):
        with open(os.path.join(ROOT, "scripts", "run_uda_sunny.ps1"),
                  encoding="utf-8-sig") as f:
            local_launcher = f.read()
        self.assertIn("$PausedMethods = @('MDD', 'MCC')", local_launcher)
        self.assertIn("$Methods.Remove($paused)", local_launcher)

        with open(os.path.join(ROOT, "scripts", "run_server.sh"),
                  encoding="utf-8") as f:
            server_launcher = f.read()
        uda_block = server_launcher.split(
            'elif [ "$PROTOCOL" = "uda" ]; then', 1
        )[1].split("\nelse\n", 1)[0]
        self.assertIn(
            'UDA_METHODS="${UDA_METHODS:-DANN,DeepCORAL,CDAN,PDA}"',
            server_launcher,
        )
        self.assertIn('run "dann_$t"', uda_block)
        self.assertIn('run "deepcoral_$t"', uda_block)
        self.assertIn('run "cdan_$t"', uda_block)
        self.assertIn('run "pda_$t"', uda_block)
        self.assertIn('if uda_method_enabled mdd', uda_block)
        self.assertIn('if uda_method_enabled mcc', uda_block)
        self.assertIn('run "mdd_$t"', uda_block)
        self.assertIn('run "mcc_$t"', uda_block)

        with open(os.path.join(ROOT, "scripts", "run_server_mdd_mcc_background.sh"),
                  encoding="utf-8") as f:
            heavy_launcher = f.read()
        self.assertIn('export UDA_METHODS="MDD,MCC"', heavy_launcher)
        self.assertIn('TARGET="${1:-backlight}"', heavy_launcher)
        self.assertIn('SEED="${2:-42}"', heavy_launcher)
        self.assertIn('JMDA_GPU_CANDIDATES:-0,1', heavy_launcher)

    def test_mdd_warmup_is_wired_through_local_and_server_launchers(self):
        with open(os.path.join(ROOT, "MDD", "train_mdd.py"), encoding="utf-8") as f:
            trainer = f.read()
        self.assertIn("--trade-off-warmup-epochs", trainer)
        self.assertIn("transfer_weight = args.trade_off * min", trainer)

        with open(os.path.join(ROOT, "scripts", "run_uda_sunny.ps1"),
                  encoding="utf-8-sig") as f:
            local_launcher = f.read()
        self.assertIn("[int]$MDDTradeOffWarmupEpochs = 0", local_launcher)
        self.assertIn("'--trade-off-warmup-epochs'", local_launcher)

        with open(os.path.join(ROOT, "scripts", "run_server.sh"),
                  encoding="utf-8") as f:
            server_launcher = f.read()
        self.assertIn(
            'UDA_MDD_TRADE_OFF_WARMUP_EPOCHS="${UDA_MDD_TRADE_OFF_WARMUP_EPOCHS:-0}"',
            server_launcher,
        )
        self.assertIn(
            '--trade-off-warmup-epochs "$UDA_MDD_TRADE_OFF_WARMUP_EPOCHS"',
            server_launcher,
        )

    def test_latest_log_driven_recipes_are_pinned_in_launchers(self):
        with open(os.path.join(ROOT, "scripts", "run_uda_tuned.ps1"),
                  encoding="utf-8-sig") as f:
            tuned = f.read()
        for expected in (
            "DANNEpochs = 22",
            "DANNGrlMaxIterations = 700",
            "DANNTargetEvalInterval = 1",
            "DeepCoralWarmupEpochs = 0",
            "DeepCoralLabelSmoothing = 0.05",
            "CdanIterations = 1400",
            "CDANCoeffMaxIterations = 12000",
            "CDANTestInterval = 100",
            "PDAEpochs = 24",
            "PDAConfidenceThreshold = 0.93",
        ):
            self.assertIn(expected, tuned)

        with open(os.path.join(ROOT, "scripts", "run_server_mdd_mcc_background.sh"),
                  encoding="utf-8") as f:
            server_heavy = f.read()
        for expected in (
            'UDA_MDD_EPOCHS:-36',
            'UDA_MDD_LR:-0.0025',
            'UDA_MDD_ITERS_PER_EPOCH:-80',
            'UDA_MDD_TRADE_OFF:-0.45',
            'UDA_MDD_TRADE_OFF_WARMUP_EPOCHS:-12',
            'UDA_MCC_EPOCHS:-26',
            'UDA_MCC_LR:-0.002',
            'UDA_MCC_ITERS_PER_EPOCH:-100',
            'UDA_MCC_TRADE_OFF:-0.15',
            'UDA_MCC_TRADE_OFF_WARMUP_EPOCHS:-12',
            'UDA_MCC_LABEL_SMOOTHING:-0.10',
        ):
            self.assertIn(expected, server_heavy)

    def test_source_label_smoothing_is_explicit_and_launcher_controlled(self):
        with open(os.path.join(ROOT, "DeepCORAL", "train_deepcoral.py"),
                  encoding="utf-8") as f:
            coral = f.read()
        self.assertIn("--label-smoothing", coral)
        self.assertIn("CrossEntropyLoss(label_smoothing=args.label_smoothing)", coral)

        with open(os.path.join(ROOT, "MCC", "train_mcc.py"),
                  encoding="utf-8") as f:
            mcc = f.read()
        self.assertIn("--label-smoothing", mcc)
        self.assertIn("label_smoothing=args.label_smoothing", mcc)

        with open(os.path.join(ROOT, "scripts", "run_uda_sunny.ps1"),
                  encoding="utf-8-sig") as f:
            launcher = f.read()
        self.assertIn("[double]$DeepCoralLabelSmoothing = 0.0", launcher)
        self.assertIn("[double]$MCCLabelSmoothing = 0.0", launcher)

    def test_local_split_launcher_uses_small_batch_and_forwards_it_to_pda(self):
        with open(os.path.join(ROOT, "scripts", "run_local_uda_four.ps1"),
                  encoding="utf-8-sig") as f:
            split_launcher = f.read()
        self.assertIn("[int]$BatchSize = 16", split_launcher)
        self.assertIn("@('DANN', 'DeepCORAL', 'CDAN', 'PDA')", split_launcher)
        self.assertIn("[int]$CdanIterations = 2000", split_launcher)
        self.assertNotIn("2000.0 * 16.0 / $BatchSize", split_launcher)

        with open(os.path.join(ROOT, "scripts", "run_uda_sunny.ps1"),
                  encoding="utf-8-sig") as f:
            local_launcher = f.read()
        self.assertIn("if ($method -eq 'PDA')", local_launcher)
        self.assertIn("'--test-batch-size', $ImageBatchSize", local_launcher)

    def test_pda_dassl_adapter_erases_target_train_labels(self):
        pda_dir = os.path.join(ROOT, "PDA")
        module = load_module(
            "jmda_official_pda_entry",
            os.path.join(pda_dir, "train_pda.py"),
            prepend=pda_dir,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "sample.png")
            Image.new("RGB", (32, 32), color=(10, 20, 30)).save(image_path)
            for name in ("sunny_train", "backlight_train", "backlight_val"):
                with open(os.path.join(temp_dir, name + ".txt"), "w", encoding="utf-8") as f:
                    f.write(f"{image_path} 0\n")
            args = argparse.Namespace(
                data_dir=temp_dir,
                target="backlight",
                num_workers=0,
                eval_num_workers=0,
                prefetch_factor=2,
                worker_timeout_seconds=300,
                persistent_workers=False,
                pin_memory=False,
                batch_size=1,
                test_batch_size=1,
                clip_cache_dir=temp_dir,
                epochs=1,
                seed=42,
            )
            cfg = module.build_cfg(args, ["Cargo Ship"], temp_dir)
            dataset = module.JMDAUda(cfg)
            self.assertEqual({item.label for item in dataset.train_u}, {-1})
            self.assertEqual({item.label for item in dataset.train_x}, {0})
            self.assertEqual({item.label for item in dataset.test}, {0})
            from dassl.data import DataManager
            manager = DataManager(cfg)
            batch_x = next(iter(manager.train_loader_x))
            batch_u = next(iter(manager.train_loader_u))
            self.assertEqual(tuple(batch_x["img"].shape), (1, 3, 224, 224))
            self.assertEqual(batch_u["label"].tolist(), [-1])

    def test_official_upstream_commits_are_pinned(self):
        with open(os.path.join(ROOT, "PDA", "official_upstream.txt"), encoding="utf-8") as f:
            pins = f.read()
        self.assertIn("21b0f4449a9e5914eba49e2415926309e742cffa", pins)
        self.assertIn("c61a1b570ac6333bd50fb5ae06aea59002fb20bb", pins)

    def test_dassl_loads_its_full_local_checkpoint_on_torch_26(self):
        dassl_root = os.path.join(ROOT, "third_party", "Dassl.pytorch")
        if dassl_root not in sys.path:
            sys.path.insert(0, dassl_root)
        from dassl.utils.torchtools import load_checkpoint

        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "full-checkpoint.pth.tar")
            torch.save({"epoch": 1, "scheduler": scheduler}, path)
            loaded = load_checkpoint(path)
        self.assertEqual(loaded["epoch"], 1)
        self.assertIsInstance(
            loaded["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR
        )

    def test_server_launcher_forwards_fast_loader_and_selection_policy(self):
        with open(os.path.join(ROOT, "scripts", "run_server.sh"), encoding="utf-8") as f:
            launcher = f.read()
        self.assertIn('UDA_NUM_WORKERS="${UDA_NUM_WORKERS:-8}"', launcher)
        self.assertIn('--num-workers "$UDA_NUM_WORKERS"', launcher)
        self.assertIn('--prefetch-factor "$UDA_PREFETCH_FACTOR"', launcher)
        self.assertIn(
            'dann_*|deepcoral_*|cdan_*|mdd_*|mcc_*|pda_*|ccsa_*|dsne_*) '
            'model_selection="best_target_val_ACC_within_budget"',
            launcher,
        )
        self.assertIn('mdd_*|mcc_*|pda_*|jmda_net_*|mcft_*) input_modalities="VIS+IR"', launcher)

        for relative in ("MDD/train_mdd.py", "MCC/train_mcc.py"):
            with open(os.path.join(ROOT, relative), encoding="utf-8") as f:
                trainer = f.read()
            self.assertIn("the highest-accuracy epoch is selected", trainer)
            self.assertIn("Selected target val metrics:", trainer)
            self.assertIn("Last-epoch target val accuracy", trainer)
            self.assertNotIn("if epoch < args.epochs:", trainer)


if __name__ == "__main__":
    unittest.main()
