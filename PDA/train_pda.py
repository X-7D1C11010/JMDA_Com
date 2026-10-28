"""Run the official PDA trainer through its official Dassl TrainerXU stack.

Only the dataset adapter is project-specific:
  * train_x: labelled sunny source images;
  * train_u: target-train images with labels replaced by -1;
  * test: labelled target-val images, used by the released PDA best-checkpoint
    process and for final ACC/Precision/Recall/F1 reporting.

The target-train labels therefore never enter the PDA optimization objective.
"""

import argparse
import os
import sys

import torch
from yacs.config import CfgNode as CN


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")
DASSL_ROOT = os.path.join(PROJECT_ROOT, "third_party", "Dassl.pytorch")
PDA_OFFICIAL_ROOT = os.path.join(PROJECT_ROOT, "PDA", "official")
for path in (PDA_OFFICIAL_ROOT, DASSL_ROOT, TOOLS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from dassl.config import get_cfg_default
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger

from class_names import load_class_names
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything
from uda_data import UdaImageListDataset, format_class_space, format_dropped, uda_label_space

# Importing the released package registers the official PDA trainer.
from trainers import PDA as OfficialPDA  # noqa: F401,E402


def _dataset_from_list(list_path, label_map):
    return UdaImageListDataset(list_path, transform=None, label_map=label_map)


@DATASET_REGISTRY.register()
class JMDAUda(DatasetBase):
    """Dassl adapter for the target-driven JMDA partial-set protocol."""

    def __init__(self, cfg):
        list_dir = cfg.DATASET.LIST_DIR
        target = cfg.DATASET.TARGET
        class_names = list(cfg.DATASET.CLASS_NAMES)
        class_ids, label_map = uda_label_space(list_dir, target)
        if len(class_names) != len(class_ids):
            raise ValueError("configured PDA class names do not match target label space")

        source_ds = _dataset_from_list(
            os.path.join(list_dir, "sunny_train.txt"), label_map
        )
        target_train_ds = _dataset_from_list(
            os.path.join(list_dir, f"{target}_train.txt"), label_map
        )
        target_val_ds = _dataset_from_list(
            os.path.join(list_dir, f"{target}_val.txt"), label_map
        )

        train_x = [
            Datum(impath=path, label=label, domain=0,
                  classname=class_names[label])
            for path, label in source_ds.samples
        ]
        # Intentionally erase target-train labels. Official PDA parse_batch_train
        # reads only batch_u["img"], and construct_bank(source=False) generates
        # CLIP pseudo-labels, so -1 is never used as a classification target.
        train_u = [
            Datum(impath=path, label=-1, domain=1, classname="")
            for path, _ in target_train_ds.samples
        ]
        target_val = [
            Datum(impath=path, label=label, domain=1,
                  classname=class_names[label])
            for path, label in target_val_ds.samples
        ]

        print(format_class_space(class_ids, len(class_ids)))
        print(f"PDA/Dassl train_x (labelled source): {len(train_x)}")
        print(f"PDA/Dassl train_u (unlabelled target, labels=-1): {len(train_u)}")
        print(f"PDA/Dassl target val (evaluation only): {len(target_val)}")
        print(format_dropped("source train", source_ds))
        super().__init__(train_x=train_x, train_u=train_u,
                         val=target_val, test=target_val)


def load_target_class_names(path, class_ids):
    if not path:
        raise ValueError(
            "PDA needs --class-names-file (or JMDA_CLASS_NAMES_FILE) with "
            "the ordered 14 real English class names"
        )
    all_names = load_class_names(path, expected_count=14)
    return [all_names[class_id] for class_id in class_ids]


def build_cfg(args, target_class_names, output_dir):
    """Released Office-31 PDA configuration plus JMDA dataset fields."""
    cfg = get_cfg_default()

    cfg.DATASET.NAME = "JMDAUda"
    cfg.DATASET.ROOT = args.data_dir
    cfg.DATASET.SOURCE_DOMAINS = ("sunny",)
    cfg.DATASET.TARGET_DOMAINS = (args.target,)
    cfg.DATASET.LIST_DIR = os.path.abspath(args.data_dir)
    cfg.DATASET.TARGET = args.target
    cfg.DATASET.CLASS_NAMES = tuple(target_class_names)

    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.TEST.NUM_WORKERS = args.eval_num_workers
    cfg.DATALOADER.PREFETCH_FACTOR = args.prefetch_factor
    cfg.DATALOADER.PERSISTENT_WORKERS = args.persistent_workers
    cfg.DATALOADER.PIN_MEMORY = args.pin_memory
    cfg.DATALOADER.WORKER_TIMEOUT_SECONDS = args.worker_timeout_seconds
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TRAIN_U.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.INTERPOLATION = "bicubic"
    cfg.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
    cfg.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
    cfg.INPUT.TRANSFORMS = ("random_resized_crop", "random_flip", "normalize")

    cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
    cfg.MODEL.BACKBONE.PATH = os.path.abspath(os.path.expanduser(args.clip_cache_dir))
    cfg.MODEL.PATCH_SIZE = 16
    cfg.MODEL.HIDDEN_SIZE = 768
    cfg.MODEL.NUM_LAYER = 12

    cfg.OPTIM.NAME = "sgd"
    cfg.OPTIM.LR = 0.003
    cfg.OPTIM.MAX_EPOCH = args.epochs
    cfg.OPTIM.LR_SCHEDULER = "cosine"
    cfg.OPTIM.WARMUP_EPOCH = 1
    cfg.OPTIM.WARMUP_TYPE = "constant"
    cfg.OPTIM.WARMUP_CONS_LR = 1e-5

    cfg.TRAIN.COUNT_ITER = "train_x"
    cfg.TRAIN.PRINT_FREQ = 10
    cfg.TRAINER.NAME = "PDA"
    cfg.TRAINER.PDA = CN()
    cfg.TRAINER.PDA.PREC = "fp16" if torch.cuda.is_available() else "fp32"
    cfg.TRAINER.PDA.DROPOUT = 0.0
    cfg.TRAINER.PDA.TP = True
    cfg.TRAINER.PDA.T_DEEP = True
    cfg.TRAINER.PDA.N_CTX = 2
    cfg.TRAINER.PDA.CSC = False
    cfg.TRAINER.PDA.CTX_INIT = "a photo of a"
    cfg.TRAINER.PDA.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.PDA.VP = True
    cfg.TRAINER.PDA.V_DEEP = True
    cfg.TRAINER.PDA.NUM_TOKENS = 2
    cfg.TRAINER.PDA.DEEP_SHARED = True
    cfg.TRAINER.PDA.DEEP_LAYERS = None
    cfg.TRAINER.PDA.SHARE_LAYER = [0, 5]
    cfg.TRAINER.PDA.LOCATION = "middle"

    cfg.CONFI = 0.9
    cfg.WARM_UP = 1
    cfg.EPOCH = 0
    cfg.DOMAINS = f"sunny-{args.target}"
    cfg.SAVE_MODEL = True
    cfg.TEST.FINAL_MODEL = "best_val"
    cfg.TEST.SPLIT = "test"
    cfg.OUTPUT_DIR = output_dir
    cfg.SEED = args.seed
    cfg.freeze()
    return cfg


@torch.no_grad()
def evaluate_selected_checkpoint(trainer, num_classes):
    trainer.set_model_mode("eval")
    labels, predictions = [], []
    for batch in trainer.test_loader:
        images, target = trainer.parse_batch_test(batch)
        output = trainer.model_inference(images)
        labels.extend(target.cpu().tolist())
        predictions.extend(output.argmax(dim=1).cpu().tolist())
    return classification_metrics(labels, predictions, num_classes=num_classes)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Official PDA + official Dassl TrainerXU on JMDA"
    )
    parser.add_argument("--target", default="backlight",
                        choices=["backlight", "rainy", "foggy", "dark"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data_lists_vis"))
    parser.add_argument("--save-dir", default=os.path.join(PROJECT_ROOT, "PDA", "output"))
    parser.add_argument("--class-names-file", default=os.environ.get("JMDA_CLASS_NAMES_FILE", ""))
    parser.add_argument("--clip-cache-dir", default=os.environ.get(
        "CLIP_CACHE_DIR", os.path.expanduser("~/.cache/clip")
    ))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--worker-timeout-seconds", type=int, default=300)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction,
                        default=torch.cuda.is_available())
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--efficiency_output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if (args.num_workers < 0 or args.eval_num_workers < 0 or
            args.prefetch_factor < 1 or args.worker_timeout_seconds < 1):
        raise ValueError("invalid DataLoader worker/prefetch settings")

    args.seed = seed_everything(args.seed)
    set_random_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    class_ids, _ = uda_label_space(args.data_dir, args.target)
    class_names = load_target_class_names(args.class_names_file, class_ids)
    output_dir = os.path.join(os.path.abspath(args.save_dir), f"pda_{args.target}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.abspath(os.path.expanduser(args.clip_cache_dir)), exist_ok=True)
    setup_logger(output_dir)

    cfg = build_cfg(args, class_names, output_dir)
    print("PDA implementation: official upstream trainer + official Dassl TrainerXU")
    print("PDA protocol: UDA (source labels + target images/pseudo-labels; "
          "target-train labels erased to -1)")
    print(f"PDA official target-best checkpoint directory: {output_dir}")
    print(f"DataLoader settings: train_workers={args.num_workers}, "
          f"eval_workers={args.eval_num_workers}, "
          f"prefetch={args.prefetch_factor}, pin_memory={args.pin_memory}, "
          f"persistent_workers={args.persistent_workers}, "
          f"timeout={args.worker_timeout_seconds}s")

    eff = None
    if args.efficiency_output_dir:
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(
            args.efficiency_output_dir, "PDA", seed=args.seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
            modalities="VIS", protocol="UDA", backbone="CLIP ViT-B/16",
            pretraining="OpenAI CLIP",
        )
        eff.epoch_start(0)

    trainer = build_trainer(cfg)
    trainer.train()
    metrics = evaluate_selected_checkpoint(trainer, num_classes=len(class_ids))

    if eff:
        eff.epoch_end(0)
        eff.finalize(extra_metrics={
            "final_acc": metrics["accuracy"],
            "target": args.target,
            "model_selection": "official_target_best",
        })

    best_path = os.path.join(output_dir, "prompt_learner", "model-best.pth.tar")
    best_epoch = "unknown"
    if os.path.isfile(best_path):
        try:
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(best_path, map_location="cpu")
        best_epoch = checkpoint.get("epoch", "unknown")
    print(format_classification_metrics(metrics))
    print(f"Best checkpoint epoch ({args.target}): {best_epoch}")
    print(f"Final target val accuracy ({args.target}, official PDA target-best checkpoint): "
          f"{metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
