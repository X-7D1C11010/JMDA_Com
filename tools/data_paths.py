# -*- coding: utf-8 -*-
"""
跨平台数据路径适配（本地 Windows ↔ Linux 服务器）。

data_lists/*.txt 在生成时写死了本机 Windows 数据根
(``d:\\Code\\JMDA-Net\\Data``)；服务器上数据根为
``/mnt/data/lixiang/data/Data``。

用法：训练前在服务器上导出环境变量即可，无需修改任何 list 文件::

    export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data

* ``remap_data_path(path)``：保留路径相对部分（含中文天气/类别目录名），
  将已知 Windows 前缀重映射到当前数据根；未设置环境变量时原样返回，
  本地 Windows 行为完全不变。
* ``data_path(*parts)``：拼接当前数据根下的文件（如 AIS .mat）。
"""
import os
import ntpath

# list 文件生成时写入的本机数据根（Windows，大小写两种写法）
_LIST_DATA_ROOTS = (
    r'd:\Code\JMDA-Net\Data',
    r'D:\Code\JMDA-Net\Data',
)


def get_data_root():
    """当前数据根目录（环境变量 JMDA_DATA_ROOT）；未设置时返回 ''。"""
    return os.environ.get('JMDA_DATA_ROOT', '').strip()


def _relative_data_parts(path):
    """Return path components relative to a known Windows data root.

    The list files always store Windows paths.  Parsing them with ``os.path``
    on Linux is incorrect because backslashes are not separators there.
    ``ntpath`` gives identical results on Windows and Linux; only the final
    join is performed with the host OS path implementation.
    """
    stored = ntpath.normcase(
        ntpath.normpath(str(path).strip().replace('/', '\\'))
    )
    for base in _LIST_DATA_ROOTS:
        base_norm = ntpath.normcase(ntpath.normpath(base))
        try:
            if ntpath.commonpath([stored, base_norm]) != base_norm:
                continue
        except ValueError:
            continue
        relative = ntpath.relpath(stored, base_norm)
        if relative == '.':
            return ()
        parts = tuple(
            part for part in relative.split('\\') if part not in ('', '.')
        )
        if parts and not any(part == '..' for part in parts):
            return parts
    return None


def remap_data_path(path):
    """将 list 中的绝对路径重映射到当前数据根；无需重映射时原样返回。"""
    root = get_data_root()
    if not root:
        return path
    parts = _relative_data_parts(path)
    if parts is None:
        return path
    return os.path.normpath(os.path.join(root, *parts))


def data_path(*parts):
    """拼接数据根下的文件路径（AIS 等）。

    有 JMDA_DATA_ROOT 时用之；否则回退到 Windows 本机数据根，保证本地
    开发默认行为不变。
    """
    root = get_data_root() or _LIST_DATA_ROOTS[0]
    return os.path.join(root, *parts)


def exact_ir_partner(vis_path):
    """Find the exact-stem IR mate across the raw train/val union.

    The dataset's original split may put VIS and IR observations in different
    phase folders.  This mirrors JMDA-Net's paired-dataset reconstruction and
    is shared by multimodal comparison methods and preflight validation.
    """
    stem = os.path.splitext(os.path.basename(vis_path))[0]
    same_split_ir = os.path.dirname(vis_path).replace('可见光', '红外')
    candidate_dirs = [same_split_ir]
    parts = same_split_ir.split(os.sep)
    for index, part in enumerate(parts):
        if part not in ('train', 'val'):
            continue
        alternate = list(parts)
        alternate[index] = 'val' if part == 'train' else 'train'
        candidate_dirs.append(os.sep.join(alternate))
        break
    extensions = ('.jpg', '.jpeg', '.png')
    for ir_dir in candidate_dirs:
        if not os.path.isdir(ir_dir):
            continue
        matches = sorted(
            filename for filename in os.listdir(ir_dir)
            if filename.lower().endswith(extensions)
            and os.path.splitext(filename)[0] == stem
        )
        if matches:
            # _image_map in JMDA-Net keeps the last lexicographic duplicate.
            return os.path.join(ir_dir, matches[-1])
    return None
