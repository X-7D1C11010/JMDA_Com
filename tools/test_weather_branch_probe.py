from copy import deepcopy
import unittest
import torch
from torch import nn
from torch.utils.data import Dataset
from weather.branch_probe import source_branch_probe


class Samples(Dataset):
    def __init__(self, partition, flip=False):
        self.partition, self.flip = partition, flip
        self.x = torch.tensor([[2., 0.], [1.8, .1], [2.1, -.1], [0., 2.], [.1, 1.8], [-.1, 2.1]])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        label = index//3
        return {"rgb": self.x[index], "nir": self.x[index]*.9, "label": 1-label if self.flip else label}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.feature_names, self.num_classes = ("VIS", "IR"), 2

    def encode(self, batch):
        self.last_features = tuple(batch[key]*self.anchor for key in ("rgb", "nir"))
        return torch.cat(self.last_features, 1)


class BranchProbeTests(unittest.TestCase):
    def test_target_validation_labels_only_score_predictions_and_model_rng_are_unchanged(self):
        model = Encoder(); state = deepcopy(model.state_dict()); rng = torch.get_rng_state()
        a = source_branch_probe(model, Samples("source_train"), Samples("target_val"), "cpu")
        b = source_branch_probe(model, Samples("source_train"), Samples("target_val", flip=True), "cpu")
        for name in ("VIS", "IR", "equal_concat"):
            self.assertEqual(a["branches"][name]["target_val_predictions"], b["branches"][name]["target_val_predictions"])
            self.assertEqual(a["branches"][name]["source_CV"]["accuracy"], 1.)
            self.assertEqual(a["branches"][name]["target_val"]["accuracy"], 1.)
            self.assertEqual(b["branches"][name]["target_val"]["accuracy"], 0.)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in state.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_disallows_target_training_and_test_partitions(self):
        for partition in ("target_train", "target_test"):
            with self.subTest(partition=partition), self.assertRaises(ValueError):
                source_branch_probe(Encoder(), Samples("source_train"), Samples(partition), "cpu")


if __name__ == "__main__":
    unittest.main()
