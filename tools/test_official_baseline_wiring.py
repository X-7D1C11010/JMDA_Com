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
        self.assertIn("resnet_name='ResNet18'", cdan)

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
        self.assertIn('mdd_*|mcc_*|pda_*) model_selection="official_target_best"', launcher)


if __name__ == "__main__":
    unittest.main()
