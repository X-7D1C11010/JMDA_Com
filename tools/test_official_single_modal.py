import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import TensorDataset


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from prepare_paired_protocol_lists import exact_ir_split  # noqa: E402
from download_official_uda_weights import WEIGHTS_BY_ID  # noqa: E402
from single_modality import (  # noqa: E402
    IR_MEAN,
    IR_STD,
    adapt_resnet_stem_to_ir,
    official_forever_domain_iterator,
    official_min_domain_loader,
)
from uda_data import UdaImageListDataset  # noqa: E402


class _AliasedStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.feature_layers = nn.Sequential(self.conv1, nn.ReLU())

    def forward(self, x):
        return self.feature_layers(x)


class OfficialSingleModalTests(unittest.TestCase):
    def test_ir_stem_is_one_channel_and_replaces_sequential_alias(self):
        model = _AliasedStem()
        expected = model.conv1.weight.detach().mean(dim=1, keepdim=True)
        adapt_resnet_stem_to_ir(model)
        self.assertEqual(model.conv1.in_channels, 1)
        self.assertIs(model.conv1, model.feature_layers[0])
        self.assertTrue(torch.allclose(model.conv1.weight, expected))
        self.assertEqual(model(torch.randn(2, 1, 8, 8)).shape, (2, 4, 8, 8))

    def test_ir_normalization_is_single_channel(self):
        self.assertEqual(len(IR_MEAN), 1)
        self.assertEqual(len(IR_STD), 1)
        self.assertGreater(IR_STD[0], 0)

    def test_dataset_loads_ir_as_one_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "ir.png"
            Image.new("L", (7, 5), color=127).save(image_path)
            list_path = root / "list.txt"
            list_path.write_text(f"{image_path} 0\n", encoding="utf-8")
            dataset = UdaImageListDataset(
                list_path, transform=lambda image: torch.from_numpy(
                    __import__("numpy").array(image)).unsqueeze(0),
                image_mode="L",
            )
            image, label = dataset[0]
            self.assertEqual(image.shape, (1, 5, 7))
            self.assertEqual(label, 0)

    def test_ir_list_keeps_unpaired_ir_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "黑天"
            for split, names in (("train", ("a", "b")), ("val", ("c",))):
                folder = root / split / "1" / "红外"
                folder.mkdir(parents=True)
                for name in names:
                    Image.new("L", (4, 4)).save(folder / f"{name}.png")
            train, stats = exact_ir_split(root, "target", "train")
            val, _ = exact_ir_split(root, "target", "val")
            self.assertEqual(len(train), 2)
            self.assertEqual(len(val), 1)
            self.assertEqual(stats["union_ir_files"], 3)

    def test_official_domain_iterators_keep_independent_batch_sizes(self):
        source = TensorDataset(torch.arange(24).view(12, 2), torch.arange(12))
        target = TensorDataset(torch.arange(16).view(8, 2), torch.arange(8))
        min_loader = official_min_domain_loader(source, target, 6, 4, 42)
        self.assertEqual(len(min_loader), 2)
        (source_batch, _), (target_batch, _) = next(iter(min_loader))
        self.assertEqual(source_batch.shape[0], 6)
        self.assertEqual(target_batch.shape[0], 4)
        forever = official_forever_domain_iterator(source, target, 3, 42)
        for _ in range(10):
            (source_batch, _), (target_batch, _) = next(forever)
            self.assertEqual(source_batch.shape[0], 3)
            self.assertEqual(target_batch.shape[0], 3)

    def test_tllib_official_budget_is_not_overwritten_for_single_modal(self):
        for relative, method in (("MDD/train_mdd.py", "MDD"),
                                 ("MCC/train_mcc.py", "MCC")):
            source = (ROOT / relative).read_text(encoding="utf-8")
            marker = f"args.iters_per_epoch = len(train_loader)"
            self.assertIn(marker, source)
            preceding = source[:source.index(marker)].rsplit("\n", 5)[-5:]
            self.assertTrue(any("if args.multimodal:" in line
                                for line in preceding), method)

    def test_launcher_exposes_both_single_modalities(self):
        source = (ROOT / "scripts" / "run_official_single_modal.ps1").read_text(
            encoding="utf-8")
        self.assertIn("@('vis', 'ir')", source)
        self.assertIn("DANNIterationsPerEpoch = 1000", source)
        self.assertIn("DANNLearningRate = 0.01", source)
        self.assertIn("CdanIterations = 100004", source)
        self.assertIn("MDDIterationsPerEpoch = 1000", source)
        self.assertIn("MCCIterationsPerEpoch = 1000", source)
        self.assertIn("--only resnet50 alexnet", source)

        server = (ROOT / "scripts" /
                  "run_server_official_single_modal.sh").read_text(
                      encoding="utf-8")
        self.assertIn("--only resnet50 alexnet", server)

    def test_single_modal_weight_inventory_is_complete(self):
        self.assertEqual(
            {"resnet50", "alexnet"} - set(WEIGHTS_BY_ID), set())
        deepcoral = (ROOT / "DeepCORAL" / "train_deepcoral.py").read_text(
            encoding="utf-8")
        self.assertIn("ensure_weight('alexnet', output_dir)", deepcoral)

    def test_dann_single_modal_path_matches_tllib_image_protocol(self):
        source = (ROOT / "DANN" / "train_dann.py").read_text(
            encoding="utf-8")
        self.assertIn("official_forever_domain_iterator", source)
        self.assertIn("joint_images = torch.cat((src_imgs, tgt_imgs), dim=0)",
                      source)
        self.assertIn("optimizer = optim.SGD", source)
        self.assertIn("nesterov=True", source)
        self.assertIn("planned_updates if args.multimodal else 1000", source)


if __name__ == "__main__":
    unittest.main()
