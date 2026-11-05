import os
import sys
import tempfile
import unittest

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from uda_multimodal import (  # noqa: E402
    CyclingDataIterator,
    FullCoverageBatchSampler,
    JmdaMultimodalEncoder,
    JmdaTllibBackbone,
    configure_jmda_visual_finetuning,
    jmda_train_transform,
    jmda_val_transform,
    label_free_target_paired_dataset,
)


class JmdaMultimodalArchitectureTests(unittest.TestCase):
    def test_direct_concatenation_matches_component_features(self):
        model = JmdaMultimodalEncoder(modality_dim=16, pretrained_vis=False).eval()
        images = torch.randn(2, 6, 64, 64)
        with torch.no_grad():
            vis, ir = model.forward_components(images)
            fused = model(images)
        self.assertEqual(tuple(vis.shape), (2, 16))
        self.assertEqual(tuple(ir.shape), (2, 16))
        self.assertEqual(model.out_features, 32)
        torch.testing.assert_close(fused, torch.cat((vis, ir), dim=1))

    def test_tllib_adapter_returns_one_by_one_feature_map(self):
        model = JmdaTllibBackbone(modality_dim=8, pretrained_vis=False).eval()
        with torch.no_grad():
            features = model(torch.randn(2, 6, 64, 64))
        self.assertEqual(tuple(features.shape), (2, 16, 1, 1))
        self.assertEqual(model.out_features, 16)

    def test_short_loader_cycles_without_shortening_source_epoch(self):
        target_loader = DataLoader(TensorDataset(torch.arange(2)), batch_size=1)
        iterator = CyclingDataIterator(target_loader)
        values = [int(next(iterator)[0].item()) for _ in range(5)]
        self.assertEqual(values, [0, 1, 0, 1, 0])

    def test_full_coverage_sampler_keeps_all_indices_without_singleton(self):
        dataset = TensorDataset(torch.arange(17))
        sampler = FullCoverageBatchSampler(dataset, batch_size=8, shuffle=False)
        batches = list(sampler)
        self.assertEqual([len(batch) for batch in batches], [8, 7, 2])
        self.assertEqual(sorted(index for batch in batches for index in batch),
                         list(range(17)))

    def test_jmda_transforms_match_paired_protocol(self):
        train_operations = jmda_train_transform().transforms
        val_operations = jmda_val_transform().transforms
        self.assertEqual(
            [type(operation) for operation in train_operations],
            [transforms.Resize, transforms.RandomCrop,
             transforms.RandomHorizontalFlip, transforms.ToTensor,
             transforms.Normalize],
        )
        self.assertEqual(train_operations[0].size, (256, 256))
        self.assertEqual(train_operations[1].size, (224, 224))
        self.assertEqual(
            [type(operation) for operation in val_operations],
            [transforms.Resize, transforms.ToTensor, transforms.Normalize],
        )
        self.assertEqual(val_operations[0].size, (224, 224))

    def test_projection_only_mode_matches_legacy_train_mode_batchnorm(self):
        model = JmdaMultimodalEncoder(modality_dim=16, pretrained_vis=False)
        details = configure_jmda_visual_finetuning(model, 'legacy_supervised')
        self.assertGreater(details['vis_trainable'], 0)
        self.assertLess(details['vis_trainable'], details['vis_total'])
        self.assertTrue(all(parameter.requires_grad
                            for parameter in model.vis_encoder.proj.parameters()))
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in model.vis_encoder.features.parameters()))
        model.train()
        for module in model.vis_encoder.features.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                self.assertTrue(module.training)

    def test_target_pair_manifest_does_not_expose_target_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vis_path = os.path.join(temp_dir, 'ship_001.png')
            ir_dir = os.path.join(temp_dir, 'ir')
            os.makedirs(ir_dir)
            ir_path = os.path.join(ir_dir, 'ship_001.png')
            Image.new('RGB', (230, 230), color='red').save(vis_path)
            Image.new('L', (230, 230), color=64).save(ir_path)
            label_list = os.path.join(temp_dir, 'dark_train.txt')
            pair_list = os.path.join(temp_dir, 'dark_train_pairs.tsv')
            with open(label_list, 'w', encoding='utf-8') as handle:
                handle.write(f'{vis_path} 7\n')
            with open(pair_list, 'w', encoding='utf-8') as handle:
                handle.write(f'{vis_path}\t{ir_path}\n')
            dataset = label_free_target_paired_dataset(
                label_list, jmda_val_transform(), {7: 0})
            inputs, sentinel = dataset[0]
        self.assertTrue(dataset.label_free)
        self.assertEqual(sentinel, -1)
        self.assertEqual(tuple(inputs.shape), (6, 224, 224))
        self.assertTrue(all(label == -1 for _, label in dataset.samples))


if __name__ == '__main__':
    unittest.main()
