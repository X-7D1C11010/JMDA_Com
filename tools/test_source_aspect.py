import unittest
import torch
from weather.source_aspect import augment_source_aspect


class AspectChecks(unittest.TestCase):
    def test_disabled_preserves_rng_and_batch(self):
        b={"rgb":torch.zeros(2,3,16,16),"nir":torch.ones(2,3,16,16)}
        state=torch.get_rng_state().clone()
        out,_=augment_source_aspect(b)
        self.assertIs(out,b);self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_paired_geometry_preserves_complete_foreground_and_labels(self):
        torch.set_num_threads(2)
        x=torch.zeros(2,3,32,32);x[:,:,8:24,3:29]=1
        labels=torch.tensor([0,1]);b={"rgb":x,"nir":x.clone(),"label":labels}
        for ratio in (.5,2.):
            out,record=augment_source_aspect(b,ratio,ratio,1.,torch.Generator().manual_seed(1))
            torch.testing.assert_close(out["rgb"],out["nir"])
            self.assertIs(out["label"],labels)
            self.assertTrue(torch.equal(x,b["rgb"]))
            self.assertGreater(float(out["rgb"].sum()),0.)
            self.assertAlmostEqual(float(out["rgb"].sum()/x.sum()),.5,places=5)
            self.assertEqual(record["source_aspect_augmented"],2)
            self.assertFalse(record["target_aspect_augmented"])


if __name__=="__main__":unittest.main()
