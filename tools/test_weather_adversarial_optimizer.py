"""Check independent adversarial optimization, parameter coverage and schedules."""
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from vds2raw.train_adaptation import make_optimizer


def model_and_args(method="cdan"):
    model=nn.Module()
    model.method=method
    model.rgb_encoder=nn.Linear(2,3)
    model.nir_encoder=nn.Linear(2,3)
    model.head=nn.Linear(6,2)
    model.modality_logits=nn.Parameter(torch.zeros(2))
    if method in {"dann","cdan"}:model.discriminator=nn.Linear(6,1)
    if method=="mdd":model.adv_head=nn.Linear(6,2)
    args=SimpleNamespace(lr=.001,ais_lr_multiplier=1.,gate_lr_multiplier=1.,gate_weight_decay=None,
        optimizer="adamw",method=method,weight_decay=0.,lr_gamma=None,lr_decay=.75,
        lr_scheduler="reference",epochs=20)
    return model,args


class AdversarialOptimizerChecks(unittest.TestCase):
    def test_disabled_keeps_existing_groups_and_rates(self):
        model,args=model_and_args()
        opt,_,_=make_optimizer(model,args)
        self.assertEqual([(x["role"],x["lr"]) for x in opt.param_groups],
                         [("pretrained_RGB_NIR",.001),("new_layers_gate",.01)])

    def test_exclusive_partition_and_scheduler_preserve_relative_learning_rates(self):
        for method in ("dann","cdan","mdd"):
            model,args=model_and_args(method)
            args.image_lr_override=.0001
            args.ir_lr_override=.0002
            args.adversarial_lr_override=.00001
            opt,schedule,_=make_optimizer(model,args)
            by_role={x["role"]:x for x in opt.param_groups}
            params=[p for group in opt.param_groups for p in group["params"]]
            self.assertEqual(len(params),len(set(map(id,params))))
            self.assertEqual(set(map(id,params)),{id(p) for p in model.parameters() if p.requires_grad})
            adversarial=getattr(model,"discriminator",getattr(model,"adv_head",None))
            self.assertEqual(set(map(id,by_role["adversarial"]["params"])),set(map(id,adversarial.parameters())))
            before={id(p):p.detach().clone() for p in model.parameters()}
            sum(p.sum() for p in model.parameters()).backward()
            opt.step()
            movement={role:float((g["params"][0]-before[id(g["params"][0])]).abs().mean())
                      for role,g in by_role.items() if g["params"]}
            self.assertLess(movement["adversarial"],movement["pretrained_RGB"]*.12)
            if schedule is not None:schedule.step()
            self.assertAlmostEqual(by_role["adversarial"]["lr"]/by_role["pretrained_RGB"]["lr"],.1)

    def test_non_adversarial_method_rejects_override(self):
        model,args=model_and_args("mcc")
        args.adversarial_lr_override=.001
        with self.assertRaisesRegex(ValueError,"requires"):make_optimizer(model,args)


if __name__=="__main__":unittest.main()
