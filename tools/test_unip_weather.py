"""Real checkpoint/gradient/phase checks for the opt-in infrared experiment."""
from pathlib import Path
import unittest
import torch
from torch import nn
from weather.unip_encoder import UNIPEncoder
from weather.encoder_schedule import lower_image_lr_after_source_warmup

ROOT = Path(__file__).resolve().parents[1]


class UNIPChecks(unittest.TestCase):
    def test_independent_ir_lr_covers_each_parameter_once_and_preserves_head_ratio(self):
        from weather.train_vds_core import parser,resolve_weather_settings
        from vds2raw.train_adaptation import make_optimizer
        m=nn.Module();m.method="deepcoral"
        m.rgb_encoder=nn.Linear(2,3);m.nir_encoder=nn.Linear(2,3)
        m.head=nn.Linear(6,2);m.modality_logits=nn.Parameter(torch.zeros(2))
        args=parser().parse_args(["--method","deepcoral","--target","dark","--data-dir",".",
            "--lr",".01","--image-lr-override",".001","--ir-lr-override",".003"])
        resolve_weather_settings(args)
        opt,schedule,_=make_optimizer(m,args)
        groups={g["role"]:g for g in opt.param_groups}
        self.assertEqual(groups["pretrained_RGB"]["lr"],.001)
        self.assertEqual(groups["pretrained_NIR"]["lr"],.003)
        self.assertEqual(groups["new_layers_gate"]["lr"],.1)
        args.optimizer="adamw"
        alternative,_,_=make_optimizer(m,args)
        self.assertIsInstance(alternative,torch.optim.AdamW)
        self.assertEqual([g['lr'] for g in alternative.param_groups],[.001,.1,.003])
        self.assertTrue(all(g['weight_decay']==args.weight_decay for g in alternative.param_groups))
        ids=[id(p) for g in opt.param_groups for p in g["params"]]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in m.parameters()})
        lower_image_lr_after_source_warmup(opt,schedule,2,1,.1)
        self.assertAlmostEqual(groups["pretrained_RGB"]["lr"],.0001)
        self.assertAlmostEqual(groups["pretrained_NIR"]["lr"],.0003)
        self.assertEqual(groups["new_layers_gate"]["lr"],.1)

    def test_released_checkpoint_heads_and_penultimate_feature_gradients(self):
        torch.set_num_threads(2)
        encoder = UNIPEncoder(ROOT/"pretrained_weights/unip/unip_vit_tiny_by_ibot_large.pth", scale="tiny")
        self.assertEqual(encoder.trunk.blocks[-1].attn.num_heads, 16)
        self.assertEqual(encoder.trunk.blocks[-1].attn.head_dim, 12)
        encoder.unfreeze_last_stage(); encoder.eval()
        x = torch.randn(2, 3, 224, 224)
        modes = encoder.extract_modes(x)
        self.assertEqual({k:tuple(v.shape) for k,v in modes.items()},
                         {"patch_mean":(2,192), "cls":(2,192), "qk_mean":(2,384), "last4":(2,768)})
        self.assertTrue(all(torch.isfinite(v).all() for v in modes.values()))
        modes["patch_mean"].square().mean().backward()
        self.assertIsNone(encoder.trunk.blocks[0].attn.qkv.weight.grad)
        self.assertIsNone(encoder.trunk.blocks[-1].attn.proj.weight.grad)
        self.assertGreater(float(encoder.trunk.blocks[-2].attn.qkv.weight.grad.norm()), 0)

    def test_phase_lr_survives_the_next_scheduler_step(self):
        p, q = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
        opt = torch.optim.SGD([{"params": [p], "lr": .01, "role": "pretrained_RGB_NIR"},
                               {"params": [q], "lr": .1, "role": "new_layers"}])
        schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:(1+.001*s)**-.75)
        lower_image_lr_after_source_warmup(opt, schedule, 2, 2, .1)
        self.assertEqual(opt.param_groups[0]["lr"], .01)
        lower_image_lr_after_source_warmup(opt, schedule, 3, 2, .1)
        opt.step(); schedule.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], .001*(1.001)**-.75)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], .1*(1.001)**-.75)

    def test_attention_pooling_keeps_released_features_and_trains_only_used_final_attention(self):
        torch.set_num_threads(2)
        encoder=UNIPEncoder(ROOT/'pretrained_weights/unip/unip_vit_tiny_by_ibot_large.pth',scale='tiny')
        encoder.eval();x=torch.rand(1,3,224,224,generator=torch.Generator().manual_seed(51))*2-1
        with torch.no_grad():before=encoder.extract_modes(x)
        encoder.feature_mode='cls_attention';encoder.unfreeze_all_used_layers()
        after=encoder.extract_modes(x)
        for key in ('patch_mean','cls','qk_mean','last4'):
            torch.testing.assert_close(before[key],after[key],atol=0,rtol=0)
        self.assertEqual(after['cls_attention'].shape,(1,192))
        self.assertEqual(encoder.output_dim,192)
        weights=encoder.last_pooling_attention
        self.assertEqual(weights.shape,(1,196));self.assertTrue(bool((weights>=0).all()))
        torch.testing.assert_close(weights.sum(1),torch.ones(1))
        after['cls_attention'].square().mean().backward()
        last=encoder.trunk.blocks[-1]
        self.assertGreater(float(last.attn.qkv.weight.grad[:384].norm()),0)
        self.assertEqual(float(last.attn.qkv.weight.grad[384:].norm()),0)
        self.assertIsNone(last.attn.proj.weight.grad)
        self.assertGreater(float(encoder.trunk.patch_embed.proj.weight.grad.norm()),0)
        self.assertFalse(encoder.trunk.pos_embed.requires_grad)

    def test_uninformative_attention_reduces_to_original_global_mean(self):
        torch.set_num_threads(2)
        encoder=UNIPEncoder(ROOT/'pretrained_weights/unip/unip_vit_tiny_by_ibot_large.pth',scale='tiny',feature_mode='cls_attention').eval()
        with torch.no_grad():
            encoder.trunk.blocks[-1].attn.qkv.weight.zero_();encoder.trunk.blocks[-1].attn.qkv.bias.zero_()
            modes=encoder.extract_modes(torch.rand(1,3,224,224,generator=torch.Generator().manual_seed(52))*2-1)
        torch.testing.assert_close(modes['cls_attention'],modes['patch_mean'],atol=1e-6,rtol=1e-6)

    def test_attention_weights_remain_normalized_under_BF16_autocast(self):
        torch.set_num_threads(2)
        encoder=UNIPEncoder(ROOT/'pretrained_weights/unip/unip_vit_tiny_by_ibot_large.pth',scale='tiny',feature_mode='cls_attention').eval()
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            output=encoder(torch.rand(1,3,224,224,generator=torch.Generator().manual_seed(53))*2-1)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(encoder.last_pooling_attention.dtype,torch.float32)
        torch.testing.assert_close(encoder.last_pooling_attention.sum(1),torch.ones(1),atol=1e-6,rtol=1e-6)


if __name__ == "__main__": unittest.main()
