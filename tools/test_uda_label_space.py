import os
import sys
import tempfile
import unittest


TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from uda_data import (  # noqa: E402
    UdaImageListDataset,
    assert_training_label_space,
    uda_label_space,
)


class UdaLabelSpaceTests(unittest.TestCase):
    @staticmethod
    def _write(root, name, rows):
        path = os.path.join(root, name)
        with open(path, "w", encoding="utf-8") as handle:
            for image, label in rows:
                handle.write(f"{image} {label}\n")
        return path

    def test_target_support_filters_source_and_uses_one_compact_map(self):
        with tempfile.TemporaryDirectory() as root:
            source_path = self._write(
                root, "sunny_train.txt",
                [("s0.jpg", 0), ("s2.jpg", 2), ("s5.jpg", 5), ("drop.jpg", 7)],
            )
            target_path = self._write(
                root, "rainy_train.txt",
                [("t0.jpg", 0), ("t2.jpg", 2), ("t5.jpg", 5)],
            )
            self._write(
                root, "rainy_val.txt",
                [("v0.jpg", 0), ("v2.jpg", 2), ("v5.jpg", 5)],
            )

            class_ids, label_map = uda_label_space(root, "rainy")
            source = UdaImageListDataset(source_path, label_map=label_map)
            target = UdaImageListDataset(target_path, label_map=label_map)

            self.assertEqual(class_ids, [0, 2, 5])
            self.assertEqual(label_map, {0: 0, 2: 1, 5: 2})
            self.assertEqual(source.skipped, 1)
            self.assertTrue(assert_training_label_space(source, target, class_ids))
            self.assertEqual({label for _, label in source.samples}, {0, 1, 2})
            self.assertEqual({label for _, label in target.samples}, {0, 1, 2})

    def test_target_train_val_support_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self._write(root, "dark_train.txt", [("a.jpg", 0), ("b.jpg", 2)])
            self._write(root, "dark_val.txt", [("c.jpg", 0), ("d.jpg", 3)])
            with self.assertRaisesRegex(ValueError, "train/val class support differs"):
                uda_label_space(root, "dark")

    def test_missing_source_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            source_path = self._write(root, "sunny_train.txt", [("s0.jpg", 0)])
            target_path = self._write(
                root, "foggy_train.txt", [("t0.jpg", 0), ("t1.jpg", 1)]
            )
            self._write(root, "foggy_val.txt", [("v0.jpg", 0), ("v1.jpg", 1)])
            class_ids, label_map = uda_label_space(root, "foggy")
            source = UdaImageListDataset(source_path, label_map=label_map)
            target = UdaImageListDataset(target_path, label_map=label_map)
            with self.assertRaisesRegex(ValueError, "label support mismatch"):
                assert_training_label_space(source, target, class_ids)


if __name__ == "__main__":
    unittest.main()
