import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from tools.multimodal_batches import batch_size, batch_slice, batch_to_device
from tools.paired_transforms import PairedNaturalImageTransform
from tools.uda_data import PairedUdaImageListDataset
from vds2raw.models import CCSAOfficeModel, DSNEOfficeModel


class _FakeResNetV2(nn.Module):
    def forward(self, images):
        pooled = images.mean(dim=(2, 3))
        return pooled.mean(dim=1, keepdim=True).expand(-1, 2048)


class MultimodalBatchTests(unittest.TestCase):
    def test_recursive_device_slice_and_size_keep_every_modality(self):
        batch = {
            "rgb": torch.randn(4, 3, 8, 8),
            "nir": torch.randn(4, 1, 8, 8),
            "ais": torch.randn(4, 13),
            "ais_mask": torch.ones(4, 13),
        }
        moved = batch_to_device(batch, torch.device("cpu"))
        sliced = batch_slice(moved, 1, 3)
        self.assertEqual(batch_size(sliced), 2)
        self.assertEqual(set(sliced), set(batch))
        self.assertEqual(sliced["rgb"].shape, (2, 3, 8, 8))
        self.assertEqual(sliced["ais"].shape, (2, 13))

    def test_weather_dataset_requires_and_returns_exact_vis_ir_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vis_dir = root / "train" / "可见光" / "Cargo"
            ir_dir = root / "train" / "红外" / "Cargo"
            vis_dir.mkdir(parents=True)
            ir_dir.mkdir(parents=True)
            vis_path = vis_dir / "scene001.png"
            ir_path = ir_dir / "scene001.png"
            Image.new("RGB", (20, 20), (10, 20, 30)).save(vis_path)
            Image.new("L", (20, 20), 40).save(ir_path)
            list_path = root / "train.txt"
            list_path.write_text(f"{vis_path} 5\n", encoding="utf-8")
            dataset = PairedUdaImageListDataset(
                list_path,
                PairedNaturalImageTransform(size=16, resize=16),
                {5: 0},
            )
            inputs, label = dataset[0]
            self.assertEqual(label, 0)
            self.assertEqual(inputs["rgb"].shape, (3, 16, 16))
            self.assertEqual(inputs["nir"].shape, (1, 16, 16))
            self.assertEqual(dataset.dropped_unpaired, 0)


class MultimodalModelTests(unittest.TestCase):
    def _inputs(self, batch=2):
        return {
            "rgb": torch.randn(batch, 3, 16, 16),
            "nir": torch.randn(batch, 1, 16, 16),
            "ais": torch.zeros(batch, 13),
            "ais_mask": torch.zeros(batch, 13),
        }

    def test_ccsa_three_modal_forward_and_mask_is_only_a_gate(self):
        model = CCSAOfficeModel(
            num_classes=3,
            pretrained=False,
            multimodal=True,
            use_ais=True,
            ais_dropout_probability=0.0,
        )
        model.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=1),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        model.eval()
        inputs = self._inputs()
        with torch.no_grad():
            first = model(inputs)
            inputs["ais_mask"] = torch.ones_like(inputs["ais_mask"])
            second = model(inputs)
        self.assertEqual(first.shape, (2, 3))
        torch.testing.assert_close(first, second)

    def test_dsne_three_modal_forward_and_old_checkpoint_is_rejected(self):
        with patch("vds2raw.models.timm.create_model", return_value=_FakeResNetV2()):
            model = DSNEOfficeModel(
                num_classes=3,
                pretrained=False,
                multimodal=True,
                use_ais=True,
                ais_dropout_probability=0.0,
            )
        model.eval()
        with torch.no_grad():
            logits, embedding = model(self._inputs(), return_embedding=True)
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(embedding.shape, (2, 512))

        state = model.state_dict()
        state.pop("nir_encoder.0.weight")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "old_single_modal.pth"
            torch.save({"model_state": state}, checkpoint)
            with self.assertRaises(RuntimeError):
                model.load_source_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
