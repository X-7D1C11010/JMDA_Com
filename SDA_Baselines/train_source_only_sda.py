#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Source-only (SDA table) — train on the labeled source domain (sunny) only.

This baseline never touches target data or labels. It is identical in the UDA
and SDA tracks; it appears in the SDA main table for reference.

Usage:
    python train_source_only_sda.py --target backlight --epochs 3
"""
import os
from sda_baselines_impl import shared_args, run


def main():
    args = shared_args('source_only')
    data_dir = args.data_dir
    train_lists = [
        ('source(sunny)', os.path.join(data_dir, 'sunny_train.txt')),
    ]
    run('source_only', args, train_lists, eff_name='Source-only-SDA')


if __name__ == '__main__':
    main()
