#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Target-only Oracle (SDA table) — train on the shared labeled target subset
only (same label budget as every SDA method). No source data is used.

Default: train on the shared SDA subset (data_lists/{target}_sda_{N}shot_seed{S}.txt),
which is the apples-to-apples oracle for the SDA label budget.
Use --full-target to train on the FULL target train set instead — this is a
reference run with a DIFFERENT label budget and must be reported separately,
never ranked together with the SDA-budget rows.

Usage:
    python train_target_only.py --target backlight --epochs 3
    python train_target_only.py --target backlight --full-target --epochs 20
"""
from sda_baselines_impl import shared_args, sda_list_path, run


def main():
    args = shared_args('target_only')
    if args.full_target:
        # full-target uses the complete target train list (different budget!)
        import os
        train_lists = [
            ('target_full', os.path.join(args.data_dir, f'{args.target}_train.txt')),
        ]
        eff_name = 'Target-only-Oracle-Full'
    else:
        train_lists = [
            ('target_sda', sda_list_path(args, args.target)),
        ]
        eff_name = 'Target-only-Oracle'
    run('target_only', args, train_lists, eff_name=eff_name)


if __name__ == '__main__':
    main()
