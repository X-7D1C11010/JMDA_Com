# -*- coding: utf-8 -*-
"""
JMDA-Net-SDA (supervised domain adaptation track) — NEW entry point; the
original main.py is NOT modified.

SDA protocol (paper main table):
  * Source labels: identical to every other SDA method (full sunny train set).
  * Target labels: the SHARED labeled subset drawn once by
    JMDA_Com/tools/make_sda_lists.py (data_lists/{target}_sda_{N}shot_seed{S}.txt).
  * Only the target classification loss L_cls_target is added on the labeled
    subset; the full JMDA-Net architecture and every alignment mechanism
    (TAL + SVD projections, OT generator, discriminator, paired sampling) is
    kept EXACTLY as in main.py.
  * Target sampling: the class-paired sampler now draws target samples ONLY
    from the labeled subset (so t_label always comes from the shared subset).
    The epoch length is source-driven (n_batches = len(src)//batch).
  * The per-epoch SVD projection update still uses the FULL (unlabeled)
    target set for feature statistics — it never reads target labels.

Usage (smoke test):
    python main_sda.py --target backlight --epochs 1 --num-iterations 1
Full run:
    python main_sda.py --target rainy
Environment overrides (same style as main.py): JMDA_DATA_ROOT, JMDA_SOURCE_ROOT,
JMDA_EFFICIENCY_OUTPUT_DIR, JMDA_TRANSPORT_MODE, JMDA_OT_*.
"""
import os
import ntpath
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import random
from datetime import datetime
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# 导入自定义模块（与 main.py 完全一致）
from DataLoad import MultiModalDomainDataset
from Tensor import TensorBasedAlignmentStable
from epoch_svd import format_svd_update, update_epoch_projections
from Discriminator import DomainDiscriminator, compute_discriminator_loss, compute_generator_loss
from Generator import NeuralOptimalTransportGenerator
from Models import VisualFeatureExtractor, IRFeatureExtractor, Classifier
from metrics_utils import MetricsHistory

# ================= SDA 协议配置 =================
# 英文目标名 -> 数据集中文目录名（与 tools/make_sda_lists.py 的命名一致）
WEATHER_MAP = {
    'backlight': '逆光',
    'rainy': '雨天',
    'foggy': '雾天',
    'dark': '黑天',
}


def parse_args():
    parser = argparse.ArgumentParser(description='JMDA-Net-SDA (supervised domain adaptation track)')
    parser.add_argument('--target', type=str, default='backlight',
                        choices=sorted(WEATHER_MAP.keys()),
                        help='Target weather (English name used by the shared SDA lists)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-iterations', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--w-cls-target', type=float, default=1.0,
                        help='Weight of the target classification loss L_cls_target')
    parser.add_argument('--sda-list-dir', type=str, default=_default_sda_list_dir())
    parser.add_argument('--labels-per-class', type=int, default=10,
                        help='SDA label budget per class (selects the shared subset list)')
    parser.add_argument('--sda-seed', type=int, default=42,
                        help='Seed used by tools/make_sda_lists.py')
    return parser.parse_args()


def _default_sda_list_dir():
    """Locate the shared SDA lists (created by JMDA_Com/tools/make_sda_lists.py).

    Priority: JMDA_SDA_LIST_DIR env > ../data_lists next to this file (this
    copy lives inside the JMDA_Com repo at JMDA-Net/) > local Windows
    JMDA_Com checkout > sibling JMDA_Com/data_lists (standalone-repo layout).
    """
    env = os.environ.get('JMDA_SDA_LIST_DIR', '').strip()
    if env:
        return env
    in_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'data_lists')
    if os.path.isdir(in_repo):
        return os.path.abspath(in_repo)
    win_path = r'd:\Code\JMDA_Com\data_lists'
    if os.path.isdir(win_path):
        return win_path
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'JMDA_Com', 'data_lists')


def _remap_data_path(path):
    """Remap list paths onto JMDA_DATA_ROOT (server: /mnt/data/lixiang/data/Data).

    Kept inline (instead of importing JMDA_Com/tools) so this repository stays
    self-contained. No-op when JMDA_DATA_ROOT is unset (local Windows default).
    """
    root = os.environ.get('JMDA_DATA_ROOT', '').strip()
    if not root:
        return path
    # Shared lists contain Windows separators even when training on Linux.
    # Parse the stored path with ntpath, then join its relative components
    # using the host OS path implementation.
    p_norm = ntpath.normcase(ntpath.normpath(str(path).replace('/', '\\')))
    for base in (r'd:\Code\JMDA-Net\Data', r'D:\Code\JMDA-Net\Data'):
        b_norm = ntpath.normcase(ntpath.normpath(base))
        try:
            if ntpath.commonpath([p_norm, b_norm]) != b_norm:
                continue
        except ValueError:
            continue
        rel = ntpath.relpath(p_norm, b_norm)
        if rel == '.':
            return os.path.normpath(root)
        rel_parts = [
            part for part in rel.split('\\') if part not in ('', '.')
        ]
        if not rel_parts or any(part == '..' for part in rel_parts):
            continue
        return os.path.normpath(os.path.join(root, *rel_parts))
    return path


def load_sda_subset(list_path):
    """Read the shared SDA list -> {norm(vis_path): original_label}."""
    subset = {}
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.rsplit(' ', 1)
            key = os.path.normcase(os.path.normpath(_remap_data_path(path)))
            subset[key] = int(label)
    return subset


def build_labeled_indices(tgt_train_ds, sda_subset):
    """Map the shared SDA list onto tgt_train_ds sample indices.

    Matching key: normalized absolute VIS path. The list label lives in the
    SOURCE global_label_map space (prepare_data.py re-encoded labels, e.g.
    folder 1 -> label 0), so the dataset's original class id is mapped through
    tgt_train_ds.label_map before comparison (same space as __getitem__).
    """
    labeled_indices = []
    for idx, sample in enumerate(tgt_train_ds.samples):
        key = os.path.normcase(os.path.normpath(sample['vis']))
        if key in sda_subset:
            mapped_label = tgt_train_ds.label_map.get(sample['label'], sample['label'])
            assert mapped_label == sda_subset[key], (
                f"Label mismatch for {sample['vis']}: dataset(mapped)={mapped_label} "
                f"list={sda_subset[key]}")
            labeled_indices.append(idx)
    matched = len(labeled_indices)
    if matched < len(sda_subset):
        print(f"[SDA] WARNING: matched {matched}/{len(sda_subset)} list entries "
              f"in the target train set ({len(sda_subset) - matched} unmatched).")
    if matched == 0:
        raise RuntimeError(
            f"No SDA subset entries matched. Check that the list paths point to "
            f"the same data root as the target dataset.")
    return labeled_indices


# ==========================================
# SDA 配对采样器（仅目标侧改动：目标样本只从标注子集中抽取）
# 源侧与 PairedClassSampler 完全一致；epoch 长度改为源驱动。
# ==========================================
class PairedClassSamplerSDA:
    def __init__(self, src_ds, tgt_ds, batch_size, labeled_indices):
        self.src_ds = src_ds
        self.tgt_ds = tgt_ds
        self.batch_size = batch_size
        self.labeled_set = set(labeled_indices)

        self.src_indices = self._build_index(src_ds, None)
        self.tgt_indices = self._build_index(tgt_ds, self.labeled_set)
        common_classes = set(self.src_indices.keys()) & set(self.tgt_indices.keys())
        self.classes = sorted(list(common_classes))
        n_labeled = len(self.labeled_set)
        print(f"SDA 配对采样器已初始化：{len(self.classes)} 个共有类别，"
              f"目标标注子集 {n_labeled} 个样本。")

    def _build_index(self, dataset, allowed):
        indices = {}
        for idx, label in enumerate(dataset.labels):
            if allowed is not None and idx not in allowed:
                continue
            indices.setdefault(label, []).append(idx)
        return indices

    def __iter__(self):
        for c in self.classes:
            random.shuffle(self.src_indices[c])
            random.shuffle(self.tgt_indices[c])

        # 源驱动 epoch：与 MDD/MCC 等 SDA 基线的循环模式一致
        n_batches = len(self.src_ds) // self.batch_size

        for _ in range(n_batches):
            batch_src_idxs = []
            batch_tgt_idxs = []
            batch_classes = random.choices(self.classes, k=self.batch_size)
            for c in batch_classes:
                s_idx = random.choice(self.src_indices[c])
                t_idx = random.choice(self.tgt_indices[c])
                batch_src_idxs.append(s_idx)
                batch_tgt_idxs.append(t_idx)
            src_batch = torch.utils.data.default_collate([self.src_ds[i] for i in batch_src_idxs])
            tgt_batch = torch.utils.data.default_collate([self.tgt_ds[i] for i in batch_tgt_idxs])
            yield src_batch, tgt_batch

    def __len__(self):
        return len(self.src_ds) // self.batch_size


class LabelSmoothingCrossEntropy(nn.Module):
    """与 main.py 完全一致。"""
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = torch.nn.functional.log_softmax(output, dim=-1)
        loss = -log_preds.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_preds, target,
                                                                                   reduction=self.reduction)


def set_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad


def configure_visual_finetuning(net_vis):
    """Unfreeze the actual ResNet layer2--4 blocks stored at indices 5--7."""
    if not hasattr(net_vis, 'features') or not isinstance(net_vis.features, nn.Sequential):
        raise TypeError("VisualFeatureExtractor must expose sequential ResNet features.")
    if len(net_vis.features) < 8 or not hasattr(net_vis, 'proj'):
        raise ValueError("Unexpected ResNet layout; cannot select layer2--4 safely.")
    set_requires_grad(net_vis, False)
    for stage_index in (5, 6, 7):
        set_requires_grad(net_vis.features[stage_index], True)
    set_requires_grad(net_vis.proj, True)


def set_frozen_batchnorm_eval(module):
    frozen_count = 0
    for child in module.modules():
        if not isinstance(child, nn.modules.batchnorm._BatchNorm):
            continue
        if not any(
            parameter.requires_grad
            for parameter in child.parameters(recurse=False)
        ):
            child.eval()
            frozen_count += 1
    return frozen_count


def main():
    # ================= 配置参数 =================
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    NUM_ITERATIONS = args.num_iterations
    W_CLS_TARGET = args.w_cls_target
    SDA_LIST_DIR = args.sda_list_dir
    LABELS_PER_CLASS = args.labels_per_class
    SDA_SEED = args.sda_seed
    EFFICIENCY_OUTPUT_DIR = os.environ.get("JMDA_EFFICIENCY_OUTPUT_DIR", "")

    # 维度（与 main.py 一致）
    VIS_DIM = 512
    IR_DIM = 512
    PROJ_DIM = 64
    FUSED_DIM = PROJ_DIM * 2
    SVD_MAX_SWEEPS = 3
    SVD_TOLERANCE = 1e-4
    TRANSPORT_MODE = os.environ.get("JMDA_TRANSPORT_MODE", "sinkhorn")
    OT_EPSILON = float(os.environ.get("JMDA_OT_EPSILON", "0.1"))
    OT_SINKHORN_ITERATIONS = int(os.environ.get("JMDA_OT_SINKHORN_ITERATIONS", "50"))
    OT_CORRECTION_SCALE = float(os.environ.get("JMDA_OT_CORRECTION_SCALE", "0.1"))
    OT_CORRECTION_REG_WEIGHT = float(os.environ.get("JMDA_OT_CORRECTION_REG_WEIGHT", "0.001"))

    print(f"设备: {DEVICE} | 模式: SDA 监督域适应 (共享目标标注子集, L_cls_target 权重={W_CLS_TARGET})")

    # ================= 数据加载 =================
    local_data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    default_data_root = (
        local_data_root if os.path.isdir(local_data_root)
        else "/mnt/data/lixiang/data/Data"
    )
    DATA_ROOT = os.environ.get("JMDA_DATA_ROOT", default_data_root)
    SOURCE_ROOT = os.environ.get(
        "JMDA_SOURCE_ROOT", os.path.join(DATA_ROOT, "晴天")
    )
    target_weather_cn = WEATHER_MAP[args.target]
    TARGET_ROOT = os.environ.get(
        "JMDA_TARGET_ROOT", os.path.join(DATA_ROOT, target_weather_cn)
    )

    # 共享 SDA 子集 list（与其他所有 SDA 方法完全相同的文件）
    sda_list_path = os.path.join(
        SDA_LIST_DIR, f'{args.target}_sda_{LABELS_PER_CLASS}shot_seed{SDA_SEED}.txt')
    assert os.path.isfile(sda_list_path), (
        f'SDA subset list missing: {sda_list_path}. '
        f'Run: python tools/make_sda_lists.py --n-per-class {LABELS_PER_CLASS} '
        f'--seed {SDA_SEED}')

    # 1. 源域训练集
    src_train_ds = MultiModalDomainDataset(SOURCE_ROOT, domain_type='source', phase='train')
    global_map = src_train_ds.get_label_map()

    # 2. 目标域训练集
    tgt_train_ds = MultiModalDomainDataset(TARGET_ROOT, domain_type='target', phase='train',
                                           global_label_map=global_map)

    src_svd_ds = MultiModalDomainDataset(
        SOURCE_ROOT, domain_type='source', phase='train',
        global_label_map=global_map, deterministic_transform=True,
    )
    tgt_svd_ds = MultiModalDomainDataset(
        TARGET_ROOT, domain_type='target', phase='train',
        global_label_map=global_map, deterministic_transform=True,
    )

    # 3. 目标域验证集
    tgt_val_ds = MultiModalDomainDataset(TARGET_ROOT, domain_type='target', phase='val',
                                         global_label_map=global_map, val_augment=False)

    # 4. 共享标注子集 -> 目标训练集索引
    sda_subset = load_sda_subset(sda_list_path)
    labeled_indices = build_labeled_indices(tgt_train_ds, sda_subset)

    print("正在检查数据集一致性...")
    print(f"  源域类别: {list(src_train_ds.label_map.keys())[:5]}...")
    print(f"  验证集类别: {list(tgt_val_ds.label_map.keys())[:5]}...")
    print(f"源域训练集: {len(src_train_ds)} | 目标域训练集: {len(tgt_train_ds)} | "
          f"标注子集: {len(labeled_indices)} | 验证集: {len(tgt_val_ds)}")
    print(f"SDA list: {sda_list_path}")

    # ================= 日志设置 =================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.environ.get(
        'JMDA_LOG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, f"sda_{args.target}_{timestamp}.log")

    logger = logging.getLogger('SDATrainingLogger')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"协议: SDA (监督域适应)")
    logger.info(f"设备: {DEVICE}")
    logger.info(f"源域: {SOURCE_ROOT}")
    logger.info(f"目标域: {TARGET_ROOT}")
    logger.info(f"共享标注子集: {sda_list_path} ({len(labeled_indices)} 样本, "
                f"每类预算 {LABELS_PER_CLASS}, seed {SDA_SEED})")
    logger.info(f"L_cls_target 权重: {W_CLS_TARGET}")
    logger.info(f"迭代次数: {NUM_ITERATIONS}")
    logger.info(f"每次迭代轮数: {EPOCHS}")
    logger.info(f"批次大小: {BATCH_SIZE}")
    logger.info(
        f"TAL投影更新: 交替SVD, 最大扫描次数={SVD_MAX_SWEEPS}, "
        f"奇异值相对收敛阈值={SVD_TOLERANCE:g}"
    )
    logger.info(
        f"传输模式={TRANSPORT_MODE}, epsilon={OT_EPSILON:g}, "
        f"Sinkhorn迭代={OT_SINKHORN_ITERATIONS}, "
        f"代价修正尺度={OT_CORRECTION_SCALE:g}"
    )
    logger.info(f"源域训练集: {len(src_train_ds)} | 目标域训练集: {len(tgt_train_ds)} | "
                f"标注子集: {len(labeled_indices)} | 验证集: {len(tgt_val_ds)}")
    logger.info(f"类别数量: {len(global_map)}")
    logger.info("=" * 80)

    # ==========================================
    # SDA 配对采样器：目标侧仅从标注子集采样
    # ==========================================
    paired_loader = PairedClassSamplerSDA(src_train_ds, tgt_train_ds, BATCH_SIZE, labeled_indices)
    val_loader = DataLoader(tgt_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            drop_last=False, num_workers=0)

    # ================= 优化器 =================
    criterion_cls = LabelSmoothingCrossEntropy(eps=0.1)
    best_val_acc = 0.0
    best_metrics_per_iteration = []

    # ================= 外层迭代循环 =================
    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()

        seed = 42 + iteration
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        print(f"\n{'='*80}")
        print(f"迭代 {iteration + 1}/{NUM_ITERATIONS}")
        print(f"随机种子: {seed}")
        print(f"{'='*80}")

        logger.info(f"\n{'='*80}")
        logger.info(f"迭代 {iteration + 1}/{NUM_ITERATIONS} 开始")
        logger.info(f"随机种子: {seed}")
        logger.info(f"{'='*80}")

        best_val_acc = 0.0
        best_val_metrics = {}

        # ================= 模型初始化（与 main.py 完全一致）=================
        net_vis = VisualFeatureExtractor(output_dim=VIS_DIM).to(DEVICE)
        net_ir = IRFeatureExtractor(output_dim=IR_DIM).to(DEVICE)

        set_requires_grad(net_ir, True)
        configure_visual_finetuning(net_vis)
        logger.info(
            "VIS fine-tuning stages=ResNet layer2--4 + projection; "
            f"trainable={sum(p.numel() for p in net_vis.parameters() if p.requires_grad):,}/"
            f"{sum(p.numel() for p in net_vis.parameters()):,}."
        )

        tal_module = TensorBasedAlignmentStable(
            input_dims=[VIS_DIM, IR_DIM],
            output_dims=[PROJ_DIM, PROJ_DIM],
            num_modalities=2,
            max_svd_sweeps=SVD_MAX_SWEEPS,
            svd_tolerance=SVD_TOLERANCE,
        ).to(DEVICE)

        generator = NeuralOptimalTransportGenerator(
            feature_dim=FUSED_DIM,
            transport_mode=TRANSPORT_MODE,
            epsilon=OT_EPSILON,
            sinkhorn_iterations=OT_SINKHORN_ITERATIONS,
            correction_scale=OT_CORRECTION_SCALE,
        ).to(DEVICE)
        discriminator = DomainDiscriminator(feature_dim=FUSED_DIM).to(DEVICE)
        classifier = Classifier(input_dim=FUSED_DIM, num_classes=len(global_map)).to(DEVICE)

        vis_params = [p for p in net_vis.parameters() if p.requires_grad]
        rest_params = [p for p in net_ir.parameters() if p.requires_grad] + \
                      list(generator.parameters()) + \
                      list(classifier.parameters())

        optimizer_g = optim.AdamW([
            {'params': vis_params, 'lr': 1e-5},
            {'params': rest_params, 'lr': 5e-4}
        ], weight_decay=1e-4)

        optimizer_d = optim.AdamW(discriminator.parameters(), lr=5e-4, weight_decay=1e-4)

        min_lr_value = 1e-6
        scheduler = ReduceLROnPlateau(optimizer_g, mode='max', factor=0.5, patience=10,
                                      min_lr=min_lr_value)
        prev_lr = optimizer_g.param_groups[0]['lr']

        # ================= 训练循环 =================
        metrics_history = MetricsHistory()
        eff = None
        if EFFICIENCY_OUTPUT_DIR:
            import sys as _sys
            # In-repo layout (JMDA-Net/ inside JMDA_Com): tools at ../tools
            _tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', 'tools')
            if not os.path.isdir(_tools_dir):
                _tools_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'JMDA_Com', 'tools')
            if not os.path.isdir(_tools_dir):
                _tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools')
            _sys.path.insert(0, _tools_dir)
            try:
                from efficiency_common import EfficiencyLogger
                eff = EfficiencyLogger(
                    EFFICIENCY_OUTPUT_DIR, 'JMDA-Net-SDA',
                    seed=seed, device=str(DEVICE), modalities='VIS+IR',
                    protocol='SDA',
                    target_label_count=len(labeled_indices),
                    target_labels_per_class=LABELS_PER_CLASS,
                    backbone='ResNet-18', pretraining='ImageNet')
                logger.info(f"Efficiency logging -> {eff.run_dir}")
            except ImportError:
                logger.warning("efficiency_common not found, skipping efficiency logging")
        for epoch in range(EPOCHS):
            start_time = time.time()
            if eff:
                eff.epoch_start(epoch)

            # --- SVD update (once per epoch) ---
            # 注意：SVD 投影更新仍使用全量目标集（不读目标标签），与 main.py 一致。
            _svd_t0 = time.perf_counter()
            svd_info = update_epoch_projections(
                tal_module=tal_module,
                encoders=[net_vis, net_ir],
                modality_keys=['vis', 'ir'],
                source_dataset=src_svd_ds,
                target_dataset=tgt_svd_ds,
                batch_size=BATCH_SIZE,
                device=DEVICE,
                seed=seed * 1000 + epoch,
                class_paired=True,
            )
            _svd_t1 = time.perf_counter()
            if eff:
                eff.record_module('svd_update', (_svd_t1 - _svd_t0) * 1000)
            logger.info(f"Epoch [{epoch + 1}/{EPOCHS}] | {format_svd_update(svd_info)}")
            epoch_projection_update_count = int(
                tal_module.projection_update_count.item()
            )

            net_vis.train()
            frozen_vis_bn_count = set_frozen_batchnorm_eval(net_vis)
            net_ir.train()
            tal_module.train()
            generator.train()
            classifier.train()
            discriminator.train()
            if epoch == 0:
                logger.info(
                    "Frozen VIS BatchNorm layers kept in eval mode: "
                    f"{frozen_vis_bn_count}."
                )

            loss_accum = 0.0
            loss_cls_accum = 0.0
            loss_cls_tgt_accum = 0.0
            loss_tal_accum = 0.0
            loss_adv_accum = 0.0
            loss_ot_reg_accum = 0.0
            marginal_residual_accum = 0.0
            row_marginal_residual_accum = 0.0
            column_marginal_residual_accum = 0.0
            ot_objective_accum = 0.0
            train_correct = 0
            train_total = 0
            steps = 0

            for src_data, tgt_data in paired_loader:
                steps += 1
                if eff:
                    eff.step_start()

                s_vis, s_ir, s_label = src_data['vis'].to(DEVICE), src_data['ir'].to(DEVICE), src_data['label'].to(DEVICE)
                t_vis, t_ir, t_label = tgt_data['vis'].to(DEVICE), tgt_data['ir'].to(DEVICE), tgt_data['label'].to(DEVICE)

                if eff:
                    eff.dataloader_done()

                # --- Forward ---
                _enc_t0 = time.perf_counter()
                f_s_vis, f_s_ir = net_vis(s_vis), net_ir(s_ir)
                f_t_vis, f_t_ir = net_vis(t_vis), net_ir(t_ir)
                _enc_t1 = time.perf_counter()
                if eff:
                    eff.record_module('encoder_forward', (_enc_t1 - _enc_t0) * 1000)

                _tal_t0 = time.perf_counter()
                (p_s_vis, p_s_ir), (p_t_vis, p_t_ir), loss_tal = tal_module(
                    [f_s_vis, f_s_ir],
                    [f_t_vis, f_t_ir],
                )
                _tal_t1 = time.perf_counter()
                if eff:
                    eff.record_module('tensor_forward', (_tal_t1 - _tal_t0) * 1000)

                if isinstance(p_s_vis, list): p_s_vis = p_s_vis[0]
                if isinstance(p_s_ir, list): p_s_ir = p_s_ir[0]
                if isinstance(p_t_vis, list): p_t_vis = p_t_vis[0]
                if isinstance(p_t_ir, list): p_t_ir = p_t_ir[0]

                feat_src = torch.cat([p_s_vis, p_s_ir], dim=1)
                feat_tgt = torch.cat([p_t_vis, p_t_ir], dim=1)

                # 生成中间域特征
                _sink_t0 = time.perf_counter()
                feat_mid, transport_details = generator(
                    feat_src,
                    feat_tgt,
                    return_details=True,
                )
                _sink_t1 = time.perf_counter()
                if eff:
                    eff.record_module('sinkhorn', (_sink_t1 - _sink_t0) * 1000)

                # --- Discriminator Step ---
                if steps % 2 == 0:
                    optimizer_d.zero_grad()
                    loss_d, _ = compute_discriminator_loss(
                        discriminator(feat_src.detach()),
                        discriminator(feat_tgt.detach()),
                        discriminator(feat_mid.detach())
                    )
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    optimizer_d.step()

                # --- Generator & Classifier Step ---
                optimizer_g.zero_grad()
                pred_src = classifier(feat_src)
                pred_tgt = classifier(feat_tgt)
                pred_mid = classifier(feat_mid)

                # SDA 损失：源分类 + 目标分类(L_cls_target, 标注子集) + 中间域分类
                # 目标侧 t_label 全部来自共享标注子集（采样器保证）。
                loss_cls_src = criterion_cls(pred_src, s_label)
                loss_cls_tgt = criterion_cls(pred_tgt, t_label)  # L_cls_target
                loss_cls_mid = criterion_cls(pred_mid, s_label)

                loss_cls_total = loss_cls_src + W_CLS_TARGET * loss_cls_tgt + loss_cls_mid

                set_requires_grad(discriminator, False)
                loss_adv = compute_generator_loss(
                    discriminator(feat_mid, use_grl=False),
                    'kl_uniform',
                )
                loss_ot_reg = transport_details['correction_regularization']

                # 损失加权与 main.py 完全一致（仅 L_cls_target 乘以 W_CLS_TARGET）
                loss_total = (
                    loss_cls_total
                    + 0.3 * loss_tal
                    + 0.15 * loss_adv
                    + OT_CORRECTION_REG_WEIGHT * loss_ot_reg
                )

                _bwd_t0 = time.perf_counter()
                loss_total.backward()
                _bwd_t1 = time.perf_counter()
                if eff:
                    eff.record_module('backward', (_bwd_t1 - _bwd_t0) * 1000)
                set_requires_grad(discriminator, True)
                torch.nn.utils.clip_grad_norm_(net_vis.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(net_ir.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                _opt_t0 = time.perf_counter()
                optimizer_g.step()
                _opt_t1 = time.perf_counter()
                if eff:
                    eff.record_module('optimizer_step', (_opt_t1 - _opt_t0) * 1000)
                    eff.step_end()

                loss_accum += loss_total.item()
                loss_cls_accum += loss_cls_total.item()
                loss_cls_tgt_accum += loss_cls_tgt.item()
                loss_tal_accum += loss_tal.item()
                loss_adv_accum += loss_adv.item()
                loss_ot_reg_accum += loss_ot_reg.item()
                marginal_residual_accum += transport_details['marginal_residual'].item()
                row_marginal_residual_accum += transport_details['row_marginal_residual'].item()
                column_marginal_residual_accum += transport_details['column_marginal_residual'].item()
                ot_objective_accum += transport_details['regularized_ot_objective'].item()

                _, predicted = torch.max(pred_tgt.data, 1)
                train_correct += (predicted == t_label).sum().item()
                train_total += t_label.size(0)

            if int(tal_module.projection_update_count.item()) != epoch_projection_update_count:
                raise RuntimeError(
                    "Tensor projections changed inside the mini-batch loop; "
                    "they must remain fixed for the complete epoch."
                )

            # --- Evaluation ---
            val_metrics = evaluate(net_vis, net_ir, tal_module, classifier, val_loader, DEVICE, global_map, epoch)
            val_metrics = metrics_history.replace_if_perfect(val_metrics, epoch)
            metrics_history.add_metrics(val_metrics)
            train_acc = train_correct / train_total if train_total > 0 else 0
            val_acc = val_metrics['accuracy']

            scheduler.step(val_acc)

            current_lr = optimizer_g.param_groups[0]['lr'] if len(optimizer_g.param_groups) > 0 else 0.0
            if abs(current_lr - prev_lr) > 1e-7:
                lr_msg = f"  >>> Learning rate reduced from {prev_lr:.6f} to {current_lr:.6f}"
                print(lr_msg)
                logger.info(lr_msg)
            prev_lr = current_lr

            avg_loss = loss_accum / steps if steps > 0 else 0.0
            avg_loss_cls = loss_cls_accum / steps if steps > 0 else 0.0
            avg_loss_cls_tgt = loss_cls_tgt_accum / steps if steps > 0 else 0.0
            avg_loss_tal = loss_tal_accum / steps if steps > 0 else 0.0
            avg_loss_adv = loss_adv_accum / steps if steps > 0 else 0.0
            avg_loss_ot_reg = loss_ot_reg_accum / steps if steps > 0 else 0.0
            avg_marginal_residual = marginal_residual_accum / steps if steps > 0 else 0.0
            avg_row_residual = row_marginal_residual_accum / steps if steps > 0 else 0.0
            avg_column_residual = column_marginal_residual_accum / steps if steps > 0 else 0.0
            avg_ot_objective = ot_objective_accum / steps if steps > 0 else 0.0

            log_msg = (f"Epoch [{epoch + 1}/{EPOCHS}] | "
                       f"Loss: {avg_loss:.4f} (Cls: {avg_loss_cls:.4f}, "
                       f"ClsTarget: {avg_loss_cls_tgt:.4f}, TAL: {avg_loss_tal:.4f}, "
                       f"Adv: {avg_loss_adv:.4f}, OT-Reg: {avg_loss_ot_reg:.6f}) | "
                       f"OT inner objective: {avg_ot_objective:.4f} | "
                       f"Marginal residual row/col/max: {avg_row_residual:.2e}/"
                       f"{avg_column_residual:.2e}/{avg_marginal_residual:.2e} | "
                       f"Train Acc: {train_acc:.4f} | "
                       f"Val Acc: {val_acc:.4f} | "
                       f"Val Precision (Macro): {val_metrics['precision_macro_present']:.4f} | "
                       f"Val Recall (Macro): {val_metrics['recall_macro_present']:.4f} | "
                       f"Val F1 (Macro): {val_metrics['f1_macro_present']:.4f} | "
                       f"Val Precision (Micro): {val_metrics['precision_micro']:.4f} | "
                       f"Val Recall (Micro): {val_metrics['recall_micro']:.4f} | "
                       f"Val F1 (Micro): {val_metrics['f1_micro']:.4f} | "
                       f"LR: {current_lr:.6f}")

            print(log_msg)
            logger.info(log_msg)

            if val_acc > best_val_acc and val_acc != 1:
                best_val_acc = val_acc
                best_val_metrics = val_metrics.copy()
                best_msg = f"  >>> New Best Val Acc: {best_val_acc:.4f}"
                print(best_msg)
                logger.info(best_msg)
                logger.info("\n混淆矩阵 (Confusion Matrix):")
                cm = val_metrics['confusion_matrix']
                classes_present = val_metrics['classes_present']
                cm_str = "\n" + "\n".join([f"  {row}" for row in cm])
                logger.info(cm_str)
                logger.info(f"矩阵形状: {cm.shape} (行=真实标签, 列=预测标签)")
                logger.info(f"验证集中出现的类别: {classes_present} (共{len(classes_present)}个类别)\n")
            if eff:
                eff.epoch_end(epoch)

        iter_time = time.time() - iter_start_time

        if best_val_acc != 1.0:
            if best_val_metrics:
                best_metrics = {
                    'acc': best_val_acc,
                    'precision_macro': best_val_metrics.get('precision_macro_present', 0.0),
                    'recall_macro': best_val_metrics.get('recall_macro_present', 0.0),
                    'f1_macro': best_val_metrics.get('f1_macro_present', 0.0),
                    'precision_micro': best_val_metrics.get('precision_micro', 0.0),
                    'recall_micro': best_val_metrics.get('recall_micro', 0.0),
                    'f1_micro': best_val_metrics.get('f1_micro', 0.0),
                    'time': iter_time
                }
            else:
                best_metrics = {
                    'acc': best_val_acc,
                    'precision_macro': 0.0,
                    'recall_macro': 0.0,
                    'f1_macro': 0.0,
                    'precision_micro': 0.0,
                    'recall_micro': 0.0,
                    'f1_micro': 0.0,
                    'time': iter_time
                }
            best_metrics_per_iteration.append(best_metrics)

        iter_msg = (f"迭代 {iteration + 1}/{NUM_ITERATIONS} 完成 | "
                    f"最佳验证准确率: {best_val_acc:.4f} | "
                    f"Precision(Macro): {best_metrics['precision_macro']:.4f} | "
                    f"Recall(Macro): {best_metrics['recall_macro']:.4f} | "
                    f"F1(Macro): {best_metrics['f1_macro']:.4f} | "
                    f"耗时: {iter_time:.2f}秒")
        print(iter_msg)
        logger.info(iter_msg)

    # ================= 所有迭代完成 =================
    logger.info("=" * 80)
    logger.info(f"所有 {NUM_ITERATIONS} 次迭代完成")
    logger.info("-" * 80)

    if eff:
        eff.finalize(extra_metrics={
            'num_iterations': NUM_ITERATIONS, 'epochs': EPOCHS,
            'protocol': 'SDA', 'target': args.target,
            'target_label_count': len(labeled_indices),
            'target_labels_per_class': LABELS_PER_CLASS,
        })

    if len(best_metrics_per_iteration) == 0:
        logger.info(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        return

    metrics_names = ['acc', 'precision_macro', 'recall_macro', 'f1_macro',
                     'precision_micro', 'recall_micro', 'f1_micro']
    metrics_stats = {}

    for metric_name in metrics_names:
        values = np.array([m[metric_name] for m in best_metrics_per_iteration])
        mean_val = np.mean(values)
        std_val = np.std(values)
        metrics_stats[metric_name] = {'mean': mean_val, 'std': std_val, 'values': values}

        logger.info(f"{metric_name}:")
        logger.info(f"  各迭代值: {[f'{v:.4f}' for v in values]}")
        logger.info(f"  均值: {mean_val:.4f}")
        logger.info(f"  标准差: {std_val:.4f}")
        logger.info("")

    logger.info(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


def evaluate(net_vis, net_ir, tal_module, classifier, dataloader, device, label_map, epoch):
    """与 main.py 的 evaluate 完全一致（不读取任何目标训练标签）。"""
    net_vis.eval()
    net_ir.eval()
    tal_module.eval()
    classifier.eval()
    correct = 0
    total = 0

    all_predicted = []
    all_labels = []

    reverse_map = {v: k for k, v in label_map.items()}
    sample_printed = False

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            vis = data['vis'].to(device)
            ir = data['ir'].to(device)
            labels = data['label'].to(device)

            f_vis, f_ir = net_vis(vis), net_ir(ir)

            p_vis = torch.mm(f_vis, tal_module.V_matrices[0])
            p_ir = torch.mm(f_ir, tal_module.V_matrices[1])

            feat = torch.cat([p_vis, p_ir], dim=1)
            outputs = classifier(feat)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if not sample_printed and epoch % 5 == 0:
                print(f"  [Val Debug] Pred: {[reverse_map.get(p.item(), str(p.item())) for p in predicted[:5]]}")
                print(f"  [Val Debug] True: {[reverse_map.get(l.item(), str(l.item())) for l in labels[:5]]}")
                sample_printed = True

    accuracy = correct / total if total > 0 else 0.0

    all_predicted = np.array(all_predicted)
    all_labels = np.array(all_labels)

    all_class_labels = sorted(list(label_map.values()))
    unique_labels_in_val = sorted(np.unique(all_labels).tolist())

    precision_macro = precision_score(all_labels, all_predicted, average='macro', zero_division=0, labels=all_class_labels)
    precision_micro = precision_score(all_labels, all_predicted, average='micro', zero_division=0, labels=all_class_labels)
    recall_macro = recall_score(all_labels, all_predicted, average='macro', zero_division=0, labels=all_class_labels)
    recall_micro = recall_score(all_labels, all_predicted, average='micro', zero_division=0, labels=all_class_labels)
    f1_macro = f1_score(all_labels, all_predicted, average='macro', zero_division=0, labels=all_class_labels)
    f1_micro = f1_score(all_labels, all_predicted, average='micro', zero_division=0, labels=all_class_labels)

    if len(unique_labels_in_val) > 0:
        precision_macro_present = precision_score(all_labels, all_predicted, average='macro', zero_division=0, labels=unique_labels_in_val)
        recall_macro_present = recall_score(all_labels, all_predicted, average='macro', zero_division=0, labels=unique_labels_in_val)
        f1_macro_present = f1_score(all_labels, all_predicted, average='macro', zero_division=0, labels=unique_labels_in_val)
    else:
        precision_macro_present = 0.0
        recall_macro_present = 0.0
        f1_macro_present = 0.0

    cm = confusion_matrix(all_labels, all_predicted, labels=all_class_labels)

    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_macro_present': precision_macro_present,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_macro_present': recall_macro_present,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'f1_macro_present': f1_macro_present,
        'f1_micro': f1_micro,
        'confusion_matrix': cm,
        'classes_present': unique_labels_in_val
    }

    return metrics


if __name__ == "__main__":
    main()
