import argparse
import importlib.util
import os
import sys
import tempfile
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(name, path, prepend=None):
    if prepend and prepend not in sys.path:
        sys.path.insert(0, prepend)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialBaselineWiringTests(unittest.TestCase):
    def test_all_multimodal_uda_methods_use_jmda_non_tensor_encoder(self):
        expected = {
            "DANN/train_dann.py": "JmdaMultimodalEncoder",
            "DeepCORAL/train_deepcoral.py": "JmdaMultimodalEncoder",
            "CDAN/train_cdan.py": "JmdaMultimodalEncoder",
            "MDD/train_mdd.py": "JmdaTllibBackbone",
            "MCC/train_mcc.py": "JmdaTllibBackbone",
        }
        for relative, symbol in expected.items():
            with self.subTest(method=relative):
                with open(os.path.join(ROOT, relative), encoding="utf-8") as f:
                    trainer = f.read()
                self.assertIn(symbol, trainer)
                self.assertNotIn("DualBackbone", trainer)

        with open(os.path.join(ROOT, "tools", "uda_multimodal.py"),
                  encoding="utf-8") as f:
            shared = f.read()
        self.assertIn("class JmdaVisualEncoder", shared)
        self.assertIn("class JmdaInfraredEncoder", shared)
        self.assertIn("torch.cat((vis_features, ir_features), dim=1)", shared)

    def test_unequal_domain_loaders_do_not_truncate_source(self):
        for relative in ("DANN/train_dann.py", "DeepCORAL/train_deepcoral.py"):
            with self.subTest(method=relative):
                with open(os.path.join(ROOT, relative), encoding="utf-8") as f:
                    trainer = f.read()
                self.assertIn("n_batches = len(src_loader)", trainer)
                self.assertIn("CyclingDataIterator(tgt_loader)", trainer)
                self.assertNotIn("n_batches = min(len(src_loader), len(tgt_loader))", trainer)

    def test_mcc_source_warmup_precedes_target_confusion(self):
        with open(os.path.join(ROOT, "MCC", "train_mcc.py"),
                  encoding="utf-8") as f:
            trainer = f.read()
        self.assertIn("source_warmup = epoch <= args.source_warmup_epochs", trainer)
        self.assertIn("transfer_weight = 0.0", trainer)
        self.assertIn("transfer_loss = mcc(y_t)", trainer)
        self.assertIn("target_conf=", trainer)

    def test_log_driven_multimodal_stabilizers_are_wired(self):
        with open(os.path.join(ROOT, "DeepCORAL", "train_deepcoral.py"),
                  encoding="utf-8") as f:
            coral = f.read()
        self.assertIn("--vis-lr-mult", coral)
        self.assertIn("--ir-lr-mult", coral)
        self.assertIn("args.vis_lr_mult * args.lr", coral)
        self.assertIn("args.ir_lr_mult * args.lr", coral)

        with open(os.path.join(ROOT, "CDAN", "train_cdan.py"),
                  encoding="utf-8") as f:
            cdan = f.read()
        self.assertIn("--source-warmup-iterations", cdan)
        self.assertIn("source_warmup = iter_num < args.source_warmup_iterations", cdan)
        self.assertIn("transfer_weight = 0.0", cdan)
        self.assertIn("CrossEntropyLoss(label_smoothing=args.label_smoothing)", cdan)

        with open(os.path.join(ROOT, "scripts", "run_uda_sunny.ps1"),
                  encoding="utf-8-sig") as f:
            local_launcher = f.read()
        for expected in (
            "'--vis-lr-mult'", "'--ir-lr-mult'",
            "'--source-warmup-iterations'", "CDANLabelSmoothing",
        ):
            self.assertIn(expected, local_launcher)

        with open(os.path.join(ROOT, "scripts", "run_server_dann_coral_cdan.sh"),
                  encoding="utf-8") as f:
            server_launcher = f.read()
        self.assertIn('--vis-lr-mult "$DEEPCORAL_VIS_LR_MULT"', server_launcher)
        self.assertIn('--source-warmup-iterations "$CDAN_SOURCE_WARMUP_ITERATIONS"',
                      server_launcher)
        self.assertIn('--model-selection best_target_val_ACC_within_budget',
                      server_launcher)

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
        self.assertIn("choices=['ResNet18', 'ResNet50', 'JMDA']", cdan)
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
            'UDA_METHODS="${UDA_METHODS:-DANN,DeepCORAL,CDAN}"',
            server_launcher,
        )
        self.assertIn('run "dann_$t"', uda_block)
        self.assertIn('run "deepcoral_$t"', uda_block)
        self.assertIn('run "cdan_$t"', uda_block)
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
            'UDA_MDD_TRADE_OFF_WARMUP_EPOCHS="${UDA_MDD_TRADE_OFF_WARMUP_EPOCHS:-16}"',
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
            "DANNLearningRate = 0.0003",
            "DANNEpochs = 30",
            "DANNDomainLossWeight = 0.25",
            "DANNGrlMaxIterations = 6000",
            "DANNLearningRateScheduler = 'constant'",
            "DANNTargetEvalInterval = 1",
            "DeepCoralLearningRate = 0.0003",
            "DeepCoralEpochs = 24",
            "DeepCoralWarmupEpochs = 0",
            "DeepCoralWeightDecay = 0.001",
            "DeepCoralLabelSmoothing = 0.10",
            "DeepCoralVisualLearningRateMultiplier = 0.1",
            "DeepCoralInfraredLearningRateMultiplier = 1.0",
            "DeepCoralMaxWeight = 0.6",
            "DeepCoralHeadLearningRateMultiplier = 3.0",
            "CdanIterations = 2000",
            "CDANHeadLearningRateMultiplier = 3.0",
            "CDANAdLearningRateMultiplier = 3.0",
            "CDANTradeOff = 0.3",
            "CDANSourceWarmupIterations = 300",
            "CDANCoeffMaxIterations = 6000",
            "CDANTestInterval = 100",
            "MDDEpochs = 32",
            "MDDIterationsPerEpoch = 60",
            "MDDTradeOff = 0.35",
            "MDDTradeOffWarmupEpochs = 16",
            "MCCLearningRate = 0.001",
            "MCCEpochs = 20",
            "MCCIterationsPerEpoch = 40",
            "MCCTradeOff = 0.01",
            "MCCTradeOffWarmupEpochs = 12",
            "MCCSourceWarmupEpochs = 2",
            "MCCLabelSmoothing = 0.15",
        ):
            self.assertIn(expected, tuned)

        with open(os.path.join(ROOT, "scripts", "run_server_mdd_mcc_background.sh"),
                  encoding="utf-8") as f:
            server_heavy = f.read()
        for expected in (
            'UDA_MDD_EPOCHS:-32',
            'UDA_MDD_LR:-0.0025',
            'UDA_MDD_ITERS_PER_EPOCH:-60',
            'UDA_MDD_TRADE_OFF:-0.35',
            'UDA_MDD_TRADE_OFF_WARMUP_EPOCHS:-16',
            'UDA_MCC_EPOCHS:-20',
            'UDA_MCC_LR:-0.001',
            'UDA_MCC_ITERS_PER_EPOCH:-40',
            'UDA_MCC_TRADE_OFF:-0.01',
            'UDA_MCC_TRADE_OFF_WARMUP_EPOCHS:-12',
            'UDA_MCC_LABEL_SMOOTHING:-0.15',
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

    def test_local_split_launcher_uses_small_batch(self):
        with open(os.path.join(ROOT, "scripts", "run_local_uda_three.ps1"),
                  encoding="utf-8-sig") as f:
            split_launcher = f.read()
        self.assertIn("[int]$BatchSize = 16", split_launcher)
        self.assertIn("@('DANN', 'DeepCORAL', 'CDAN')", split_launcher)
        self.assertIn("[int]$CdanIterations = 2000", split_launcher)
        self.assertNotIn("2000.0 * 16.0 / $BatchSize", split_launcher)

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
            'dann_*|deepcoral_*|cdan_*|mdd_*|mcc_*) '
            'model_selection="best_target_val_ACC_within_budget"',
            launcher,
        )
        self.assertIn('mdd_*|mcc_*|jmda_net_*|mcft_*) input_modalities="VIS+IR"', launcher)

    def test_unified_server_launcher_runs_all_five_sequentially(self):
        with open(os.path.join(
                ROOT, "scripts", "run_server_all_uda_background.sh"),
                encoding="utf-8") as handle:
            unified = handle.read()
        self.assertIn(
            'export UDA_METHODS="DANN,DeepCORAL,CDAN,MDD,MCC"', unified)
        self.assertIn(
            'DANN -> DeepCORAL -> CDAN -> MDD -> MCC', unified)
        self.assertIn('export JMDA_GPU_SCHEDULER=1', unified)
        self.assertIn('exec bash "$HERE/scripts/run_server_background.sh" uda',
                      unified)

        with open(os.path.join(ROOT, "scripts", "run_server.sh"),
                  encoding="utf-8") as handle:
            server = handle.read()
        order = [
            server.index('if uda_method_enabled dann; then',
                         server.index('elif [ "$PROTOCOL" = "uda" ]')),
            server.index('if uda_method_enabled deepcoral; then',
                         server.index('elif [ "$PROTOCOL" = "uda" ]')),
            server.index('if uda_method_enabled cdan; then',
                         server.index('elif [ "$PROTOCOL" = "uda" ]')),
            server.index('if uda_method_enabled mdd; then',
                         server.index('elif [ "$PROTOCOL" = "uda" ]')),
            server.index('if uda_method_enabled mcc; then',
                         server.index('elif [ "$PROTOCOL" = "uda" ]')),
        ]
        self.assertEqual(order, sorted(order))
        self.assertIn('--domain-loss-weight "$UDA_DANN_DOMAIN_LOSS_WEIGHT"',
                      server)
        self.assertIn('--coral-max-weight "$UDA_DEEPCORAL_MAX_WEIGHT"',
                      server)
        self.assertIn('--ad-lr-mult "$UDA_CDAN_AD_LR_MULT"', server)

        for relative in ("MDD/train_mdd.py", "MCC/train_mcc.py"):
            with open(os.path.join(ROOT, relative), encoding="utf-8") as f:
                trainer = f.read()
            self.assertIn("the highest-accuracy epoch is selected", trainer)
            self.assertIn("Selected target val metrics:", trainer)
            self.assertIn("Last-epoch target val accuracy", trainer)
            self.assertNotIn("if epoch < args.epochs:", trainer)

    def test_multimodal_baselines_share_jmda_data_and_finetuning_protocol(self):
        for relative in (
            "DANN/train_dann.py",
            "DeepCORAL/train_deepcoral.py",
            "CDAN/train_cdan.py",
            "MDD/train_mdd.py",
            "MCC/train_mcc.py",
        ):
            with open(os.path.join(ROOT, relative), encoding="utf-8") as f:
                trainer = f.read()
            self.assertIn("label_free_target_paired_dataset", trainer)
            self.assertIn("jmda_train_transform", trainer)
            self.assertIn("jmda_val_transform", trainer)
            self.assertIn("configure_jmda_visual_finetuning", trainer)
            self.assertIn("legacy train-mode BN", trainer)


if __name__ == "__main__":
    unittest.main()
