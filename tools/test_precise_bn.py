import unittest
import torch
from torch import nn
from torch.utils.data import Dataset
from weather.precise_bn import refit_population_bn,refit_population_bn_layerwise


class View(Dataset):
    def __init__(self, partition="target_train", labelled=False):
        self.partition, self.labelled = partition, labelled
        self.values = torch.arange(5 * 2 * 3 * 3).reshape(5, 2, 3, 3).float() / 10

    def __len__(self): return len(self.values)

    def __getitem__(self, index):
        item = {"rgb": self.values[index]}
        if self.labelled: item["label"] = index % 2
        return item


class PreciseBNChecks(unittest.TestCase):
    def test_layerwise_fitting_matches_final_inputs_of_every_layer(self):
        model=nn.Sequential(nn.BatchNorm2d(2),nn.ReLU(),nn.Conv2d(2,3,1),nn.BatchNorm2d(3)).eval()
        view=View()
        before=[parameter.detach().clone() for parameter in model.parameters()]
        record=refit_population_bn_layerwise(model,view,"rgb","cpu",batch_size=2)
        measured={}
        hooks=[layer.register_forward_pre_hook(lambda _m,inputs,key=name: measured.setdefault(key,[]).append(inputs[0].detach()))
               for name,layer in model.named_modules() if isinstance(layer,nn.BatchNorm2d)]
        with torch.no_grad():model(view.values)
        for hook in hooks:hook.remove()
        for name,layer in model.named_modules():
            if not isinstance(layer,nn.BatchNorm2d):continue
            inputs=torch.cat(measured[name])
            torch.testing.assert_close(layer.running_mean,inputs.mean((0,2,3)))
            torch.testing.assert_close(layer.running_var,inputs.var((0,2,3),correction=0))
        for old,new in zip(before,model.parameters()):torch.testing.assert_close(old,new,rtol=0,atol=0)
        self.assertEqual(record["BN_layers"],2)
        with self.assertRaises(ValueError):refit_population_bn_layerwise(model,View("target_val",True),"rgb","cpu")

    def test_all_pixel_moments_independent_of_batch_size_without_parameter_updates(self):
        torch.set_num_threads(2)
        view = View()
        actual = []
        for batch in (2, 4):
            model = nn.Sequential(nn.BatchNorm2d(2), nn.ReLU()).eval()
            parameters = [p.detach().clone() for p in model.parameters()]
            record = refit_population_bn(model, view, "rgb", "cpu", batch_size=batch)
            mean = view.values.mean((0, 2, 3))
            variance = view.values.var((0, 2, 3), correction=0)
            torch.testing.assert_close(model[0].running_mean, mean)
            torch.testing.assert_close(model[0].running_var, variance)
            for left, right in zip(parameters, model.parameters()): torch.testing.assert_close(left, right, rtol=0, atol=0)
            actual.append(model(view.values).detach())
            self.assertEqual(record["fit_samples"], 5)
            self.assertFalse(model.training)
            self.assertEqual(len(model[0]._forward_pre_hooks), 0)
        torch.testing.assert_close(*actual)

    def test_validation_test_and_labelled_target_rejected_before_forward(self):
        for partition, labelled in (("target_val", True), ("target_test", True), ("target_train", True)):
            model = nn.Sequential(nn.BatchNorm2d(2)).eval()
            with self.assertRaises(ValueError): refit_population_bn(model, View(partition, labelled), "rgb", "cpu")
            torch.testing.assert_close(model[0].running_mean, torch.zeros(2))

    def test_injected_target_label_fails_and_leaves_buffers_unchanged(self):
        view = View(labelled=True)
        view.labelled = False
        original_getitem = View.__getitem__
        class Poisoned(View):
            def __getitem__(self, index):
                item = original_getitem(self, index)
                item["label"] = 999
                return item
        model = nn.Sequential(nn.BatchNorm2d(2)).eval()
        with self.assertRaises(RuntimeError): refit_population_bn(model, Poisoned(), "rgb", "cpu")
        torch.testing.assert_close(model[0].running_var, torch.ones(2))
        self.assertEqual(len(model[0]._forward_pre_hooks), 0)


if __name__ == "__main__": unittest.main()
