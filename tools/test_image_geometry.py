import unittest

import numpy as np
from PIL import Image

from weather.image_geometry import LetterboxSquare


class ImageGeometryTests(unittest.TestCase):
    def test_wide_shape_is_preserved_without_changing_input_or_network_shape(self):
        image = Image.new("RGB", (48, 12), (255, 0, 0))
        before = image.tobytes()
        result = LetterboxSquare(24, (128, 128, 128))(image)
        self.assertEqual(result.size, (24, 24))
        self.assertEqual(image.tobytes(), before)
        values = np.asarray(result)
        red = values[:, :, 0] == 255
        self.assertEqual(int(red.any(1).sum()), 6)
        self.assertEqual(int(red.any(0).sum()), 24)

    def test_square_identity_preserves_pixels_and_grayscale_channels(self):
        values = np.arange(64, dtype=np.uint8).reshape(8, 8)
        image = Image.fromarray(values).convert("RGB")
        result = LetterboxSquare(8, (128, 128, 128))(image)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(image))


if __name__ == "__main__":
    unittest.main()
