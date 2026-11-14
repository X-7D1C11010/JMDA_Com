import unittest
import torch
from weather.source_zoom import augment_source_zoom


class SourceZoomTests(unittest.TestCase):
    def test_paired_geometry_source_labels_and_training_rng_are_preserved(self):
        torch.manual_seed(42)
        rgb = torch.randn(4, 3, 24, 24)
        source = {"rgb": rgb, "nir": 2*rgb+1, "label": torch.arange(4)}
        original = rgb.clone()
        state = torch.get_rng_state()
        result, record = augment_source_zoom(source, .25, 1., torch.Generator().manual_seed(99))
        self.assertEqual(record["source_zoom_applied_samples"], 4)
        self.assertIs(result["label"], source["label"])
        torch.testing.assert_close(result["nir"], 2*result["rgb"]+1, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(source["rgb"], original, rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(torch.equal(result["rgb"], original))

    def test_disabled_path_is_an_exact_identity_without_random_draws(self):
        source = {"rgb": torch.randn(2, 3, 16, 16), "nir": torch.randn(2, 3, 16, 16), "label": object()}
        generator = torch.Generator().manual_seed(5)
        before = generator.get_state()
        self.assertIs(augment_source_zoom(source, 1., generator=generator)[0], source)
        self.assertTrue(torch.equal(before, generator.get_state()))
        with self.assertRaisesRegex(ValueError, "dedicated"):
            augment_source_zoom(source, .25)


if __name__ == "__main__":
    unittest.main()
