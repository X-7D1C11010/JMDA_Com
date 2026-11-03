import os.path as osp
import datetime
import time

import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.utils import MetricMeter, AverageMeter, load_checkpoint, save_checkpoint

from utils.clip_part import *
from utils.templates import CUSTOM_TEMPLATES


class Base_PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        pass

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        '''
        dim0 is either batch_size (during training) or n_cls (during testing)
        ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        '''
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx.half(),  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        if hasattr(self, 'dropout') and self.dropout != None:
            prompts = self.dropout(prompts)
            
        return prompts

    def forward(self):
        pass


class Base_CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.text_encoder = Simple_TextEncoder(clip_model)

        self.image_encoder = clip_model.visual
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts = clip.tokenize(prompts)

    def forward(self, image):
        text_features = self.text_encoder(self.tokenized_prompts.to(self.logit_scale.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t().cuda(image_features.device)

        return logits


@TRAINER_REGISTRY.register()
class BaseDA(TrainerXU):
    def check_cfg(self, cfg):
        trainer = cfg.TRAINER.NAME.split('_')[0].upper()
        assert cfg.TRAINER[trainer].PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        pass
    
    def train(self):
        """Generic training loops."""

        self.before_train()
        self._early_stop = False
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            if self._early_stop:
                break
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError('Training batch name is wrong!')

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 \
                or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch -
                              1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                          self.epoch + 1,
                          self.max_epoch,
                          self.batch_idx + 1,
                          self.num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          eta=eta,
                          losses=losses,
                          lr=self.get_current_lr(),
                      ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch_x, batch_u):
        pass
    
    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x["img"]
        label = batch_x["label"]
        input_u = batch_u["img"]
        input = input.to(self.device)
        label = label.to(self.device)
        input_u = input_u.to(self.device)
        return input, label, input_u
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        diagnostic_interval = getattr(
            self.cfg.TRAIN, "DIAGNOSTIC_TARGET_EVAL_INTERVAL", 0
        )
        diagnostic_eval = (
            do_test and diagnostic_interval > 0 and
            ((self.epoch + 1) % diagnostic_interval == 0 or last_epoch)
        )
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if diagnostic_eval and self.cfg.TEST.FINAL_MODEL != "best_val":
            curr_result = self.test()
            if not hasattr(self, "diagnostic_best_result"):
                self.diagnostic_best_result = -float("inf")
                self.diagnostic_best_epoch = 0
                self.diagnostic_best_metrics = None
                self.diagnostic_best_labels = None
                self.diagnostic_best_predictions = None
            current_metrics = dict(getattr(self, "last_evaluation_results", {}))
            current_f1 = current_metrics.get("macro_f1", -float("inf"))
            best_f1 = (
                self.diagnostic_best_metrics.get("macro_f1", -float("inf"))
                if self.diagnostic_best_metrics is not None else -float("inf")
            )
            # Evaluators report percentages as floats.  The same number of
            # correct samples can differ by a few ulps, so exact equality can
            # incorrectly bypass the macro-F1 tie-break (for example 27/36 at
            # two epochs).  Treat numerically equal ACC values as a tie.
            acc_tolerance = 1e-9
            improved_acc = curr_result > (
                self.diagnostic_best_result + acc_tolerance
            )
            tied_acc_better_f1 = (
                abs(curr_result - self.diagnostic_best_result) <= acc_tolerance
                and current_f1 > best_f1
            )
            if improved_acc or tied_acc_better_f1:
                self.diagnostic_best_result = curr_result
                self.diagnostic_best_epoch = self.epoch + 1
                self.diagnostic_best_metrics = current_metrics
                self.diagnostic_best_labels = list(
                    getattr(self, "last_evaluation_labels", [])
                )
                self.diagnostic_best_predictions = list(
                    getattr(self, "last_evaluation_predictions", [])
                )
            print(
                "Target accuracy at epoch {}: {:.4f}; selected peak {:.4f} "
                "at epoch {}".format(
                    self.epoch + 1, curr_result,
                    self.diagnostic_best_result, self.diagnostic_best_epoch,
                )
            )
            self.set_model_mode("train")

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test()
            patience = getattr(self.cfg.TRAIN, "EARLY_STOP_PATIENCE", 0)
            min_delta = getattr(self.cfg.TRAIN, "EARLY_STOP_MIN_DELTA", 0.0)
            if not hasattr(self, "_epochs_without_improvement"):
                self._epochs_without_improvement = 0
            is_best = curr_result > self.best_result + min_delta
            if is_best:
                self.best_result = curr_result
                self._epochs_without_improvement = 0
                if self.save:
                    self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")
            else:
                self._epochs_without_improvement += 1
            if patience > 0 and self._epochs_without_improvement >= patience:
                print(
                    "Early stopping at epoch {}; best target val result={:.4f}".format(
                        self.epoch + 1, self.best_result
                    )
                )
                self._early_stop = True
            self.set_model_mode("train")
        
        if self.save and (meet_checkpoint_freq or last_epoch or self._early_stop):
            self.save_model(self.epoch, self.output_dir)
             
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        self.last_evaluation_results = dict(results)
        self.last_evaluation_labels = list(
            getattr(self.evaluator, "_y_true", [])
        )
        self.last_evaluation_predictions = list(
            getattr(self.evaluator, "_y_pred", [])
        )
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all
     
    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )
            
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

