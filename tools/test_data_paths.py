import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_paths import _relative_data_parts, exact_ir_partner, remap_data_path


class CrossPlatformDataPathTests(unittest.TestCase):
    def test_windows_list_path_is_split_independently_of_host_os(self):
        parts = _relative_data_parts(
            r'd:\Code\JMDA-Net\Data\逆光\train\1\可见光\006.jpg'
        )
        self.assertEqual(
            parts,
            ('逆光', 'train', '1', '可见光', '006.jpg'),
        )

    def test_forward_slashes_are_also_accepted(self):
        parts = _relative_data_parts(
            'D:/Code/JMDA-Net/Data/雨天/val/2/可见光/010.jpg'
        )
        self.assertEqual(
            parts,
            ('雨天', 'val', '2', '可见光', '010.jpg'),
        )

    def test_different_root_is_not_silently_remapped(self):
        path = r'E:\other\dataset\sample.jpg'
        self.assertIsNone(_relative_data_parts(path))
        with mock.patch.dict(os.environ, {'JMDA_DATA_ROOT': '/srv/data'},
                             clear=False):
            self.assertEqual(remap_data_path(path), path)

    def test_relative_components_are_joined_under_runtime_root(self):
        runtime_root = os.path.join(os.sep, 'mnt', 'data', 'lixiang', 'data', 'Data')
        with mock.patch.dict(os.environ, {'JMDA_DATA_ROOT': runtime_root},
                             clear=False):
            mapped = remap_data_path(
                r'd:\Code\JMDA-Net\Data\雾天\train\3\可见光\x.jpg'
            )
        self.assertEqual(
            os.path.normpath(mapped),
            os.path.normpath(
                os.path.join(runtime_root, '雾天', 'train', '3', '可见光', 'x.jpg')
            ),
        )

    def test_exact_ir_partner_searches_the_other_raw_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vis_dir = root / 'train' / '1' / '可见光'
            ir_dir = root / 'val' / '1' / '红外'
            vis_dir.mkdir(parents=True)
            ir_dir.mkdir(parents=True)
            vis_path = vis_dir / 'observation.jpg'
            ir_path = ir_dir / 'observation.png'
            vis_path.write_bytes(b'vis')
            ir_path.write_bytes(b'ir')
            self.assertEqual(
                os.path.normpath(exact_ir_partner(str(vis_path))),
                os.path.normpath(str(ir_path)),
            )


if __name__ == '__main__':
    unittest.main()
