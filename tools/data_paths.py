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

# list 文件生成时写入的本机数据根（Windows，大小写两种写法）
_LIST_DATA_ROOTS = (
    r'd:\Code\JMDA-Net\Data',
    r'D:\Code\JMDA-Net\Data',
)


def get_data_root():
    """当前数据根目录（环境变量 JMDA_DATA_ROOT）；未设置时返回 ''。"""
    return os.environ.get('JMDA_DATA_ROOT', '').strip()


def remap_data_path(path):
    """将 list 中的绝对路径重映射到当前数据根；无需重映射时原样返回。"""
    root = get_data_root()
    if not root:
        return path
    p_norm = os.path.normcase(os.path.normpath(path))
    for base in _LIST_DATA_ROOTS:
        b_norm = os.path.normcase(os.path.normpath(base))
        if p_norm == b_norm or p_norm.startswith(b_norm + os.sep):
            rel = os.path.relpath(p_norm, b_norm)
            return os.path.normpath(os.path.join(root, rel))
    return path


def data_path(*parts):
    """拼接数据根下的文件路径（AIS 等）。

    有 JMDA_DATA_ROOT 时用之；否则回退到 Windows 本机数据根，保证本地
    开发默认行为不变。
    """
    root = get_data_root() or _LIST_DATA_ROOTS[0]
    return os.path.join(root, *parts)
