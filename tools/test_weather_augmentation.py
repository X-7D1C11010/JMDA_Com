import unittest
import torch
from weather.weather_augmentation import augment_source_weather


class WeatherAugmentationChecks(unittest.TestCase):
    def test_disabled_variant_is_exact_identity_without_random_draws(self):
        x={"rgb":torch.randn(2,3,16,16),"nir":torch.randn(2,3,16,16),"label":torch.tensor([0,1])}
        state=torch.get_rng_state().clone()
        y,record=augment_source_weather(x,"dark",0.)
        self.assertIs(y,x);self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(record["source_weather_augmented"],0)

    def test_source_only_augmentation_is_reproducible_finite_and_preserves_label_and_original(self):
        torch.set_num_threads(2)
        raw=torch.rand(3,3,32,32)
        mean=raw.new_tensor([.485,.456,.406]).view(1,3,1,1)
        std=raw.new_tensor([.229,.224,.225]).view(1,3,1,1)
        x={"rgb":(raw-mean)/std,"nir":torch.rand(3,3,32,32)*2-1,"label":torch.tensor([0,1,2]),"path":["a","b","c"]}
        original=x["rgb"].clone()
        for weather in ("dark","rainy","foggy"):
            a,record=augment_source_weather(x,weather,1.,torch.Generator().manual_seed(42))
            b,_=augment_source_weather(x,weather,1.,torch.Generator().manual_seed(42))
            torch.testing.assert_close(a["rgb"],b["rgb"])
            self.assertIs(a["label"],x["label"])
            self.assertTrue(torch.equal(original,x["rgb"]))
            self.assertTrue(torch.isfinite(a["rgb"]).all())
            self.assertGreater(float((a["rgb"]-x["rgb"]).abs().sum()),0)
            self.assertEqual(record["source_weather_augmented"],3)
            self.assertFalse(record["target_weather_augmented"])


if __name__=="__main__":unittest.main()
