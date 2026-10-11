#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Joint Training (SDA table) — CE on labeled source + CE on the shared labeled
target subset. Contains NO domain-alignment module of any kind (no adversarial
loss, no feature alignment, no pseudo-labels). This isolates the contribution
of the labeled target subset from that of domain alignment.

The target subset is exactly the same shared SDA list used by every other SDA
method (data_lists/{target}_sda_{N}shot_seed{S}.txt).

Usage:
    python train_joint_training.py --target backlight --epochs 3
"""
import os
from sda_baselines_impl import shared_args, sda_list_path, run


def main():
    args = shared_args('joint')
    data_dir = args.data_dir
    train_lists = [
        ('source(sunny)', os.path.join(data_dir, 'sunny_train.txt')),
        ('target_sda', sda_list_path(args, args.target)),
    ]
    run('joint', args, train_lists, eff_name='Joint-Training')


if __name__ == '__main__':
    main()
