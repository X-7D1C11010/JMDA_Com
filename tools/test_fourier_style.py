import unittest
import numpy as np
import torch
from weather.fourier_style import replace_low_frequency_amplitude, augment_source_style


class FourierStyleTests(unittest.TestCase):
    def test_numpy_reference_amplitude_replacement_and_disabled_identity(self):
        rng = np.random.default_rng(42)
        x, y = (rng.random((2, 3, 16, 16)).astype(np.float32) for _ in range(2))
        z = np.fft.fft2(x)
        a = np.fft.fftshift(np.abs(z), axes=(-2, -1))
        b = np.fft.fftshift(np.abs(np.fft.fft2(y)), axes=(-2, -1))
        a[..., 7:10, 7:10] = b[..., 7:10, 7:10]
        expected = np.fft.ifft2(np.fft.ifftshift(a, axes=(-2, -1))*np.exp(1j*np.angle(z))).real
        actual = replace_low_frequency_amplitude(torch.from_numpy(x), torch.from_numpy(y), .1)
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6, rtol=2e-6)
        original = torch.from_numpy(x)
        self.assertIs(replace_low_frequency_amplitude(original, torch.from_numpy(y), 0.), original)

    def test_source_labels_target_tensors_and_training_rng_are_preserved(self):
        torch.manual_seed(42)
        source = {"rgb": torch.randn(4, 3, 16, 16), "nir": torch.randn(4, 3, 16, 16), "label": torch.arange(4)}
        target = {"rgb": torch.randn(4, 3, 16, 16), "nir": torch.randn(4, 3, 16, 16), "label": object()}
        unchanged = {key: value.clone() for key, value in target.items() if torch.is_tensor(value)}
        generator = torch.Generator().manual_seed(8)
        state = torch.get_rng_state()
        result, record = augment_source_style(source, target, .05, 1., generator)
        self.assertIs(result["label"], source["label"])
        self.assertEqual(record["FDA_applied_samples"], 4)
        self.assertEqual(set(record["FDA_pixel_statistics"]), {"rgb", "nir"})
        for stats in record["FDA_pixel_statistics"].values():
            self.assertTrue(0 <= stats["clipped_fraction"] <= 1)
            self.assertTrue(0 <= stats["styled_mean"] <= 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        for key, value in unchanged.items():
            torch.testing.assert_close(target[key], value, rtol=0, atol=0)
        self.assertIs(augment_source_style(source, target, 0.)[0], source)


if __name__ == "__main__":
    unittest.main()
