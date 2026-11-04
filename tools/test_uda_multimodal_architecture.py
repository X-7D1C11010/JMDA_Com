import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from uda_multimodal import (  # noqa: E402
    CyclingDataIterator,
    FullCoverageBatchSampler,
    JmdaMultimodalEncoder,
    JmdaTllibBackbone,
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


if __name__ == '__main__':
    unittest.main()
