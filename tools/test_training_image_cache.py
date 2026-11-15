from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from tools.test_weather_vds_core import fixture
from tools.uda_multimodal import UdaPairedImageListDataset
from weather.train_vds_core import prepare_datasets
from weather.training_image_cache import cache_training_resizes


class TrainingCacheChecks(unittest.TestCase):
    def test_cached_resize_preserves_pixels_labels_synchronized_geometry_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);lists=fixture(root)
            source,target,*_=prepare_datasets(lists,"backlight",True)[0]
            for view in (source,target):
                raw=view.dataset
                for row in raw.paired_samples:
                    grid=np.arange(79*317*3,dtype=np.uint32).reshape(79,317,3)
                    Image.fromarray((grid%251).astype(np.uint8)).save(row[0])
                    ir=np.arange(93*271,dtype=np.uint32).reshape(93,271)
                    Image.fromarray((ir%253).astype(np.uint8)).save(row[1])
                reference=deepcopy(raw)
                state=UdaPairedImageListDataset._rng_state()
                report=cache_training_resizes(raw)
                cache_state=UdaPairedImageListDataset._rng_state()
                self.assertTrue(torch.equal(state[2],cache_state[2]))
                self.assertEqual(state[0],cache_state[0])
                np.testing.assert_equal(state[1],cache_state[1])
                self.assertEqual(report["samples"],len(raw))
                for seed in (7,42,88):
                    torch.manual_seed(seed)
                    start=UdaPairedImageListDataset._rng_state()
                    expected,label=reference[0]
                    expected_end=UdaPairedImageListDataset._rng_state()
                    UdaPairedImageListDataset._set_rng_state(start)
                    actual,actual_label=raw[0]
                    actual_end=UdaPairedImageListDataset._rng_state()
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    self.assertEqual(actual_label,label)
                    self.assertTrue(torch.equal(expected_end[2],actual_end[2]))
                    self.assertEqual(expected_end[0],actual_end[0])
                    np.testing.assert_equal(expected_end[1],actual_end[1])
                if not view.labelled:self.assertEqual(actual_label,-1)

    def test_validation_resize_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            lists=fixture(Path(directory))
            validation=prepare_datasets(lists,"backlight",True)[0][-1]
            with self.assertRaises(ValueError):cache_training_resizes(validation.dataset)


if __name__=="__main__":unittest.main()
