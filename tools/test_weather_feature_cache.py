from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import Dataset
from weather.feature_cache import cache_selected_features
from weather.infrared_polarity import augment_source_ir_polarity
from tools import test_crossmodal_loss as loss_checks


class View(Dataset):
    def __init__(self,partition,poison=False):
        self.partition=partition;self.labelled=partition!="target_train";self.poison=poison
    def __len__(self):return 6
    def __getitem__(self,index):
        image=torch.full((3,8,8),index/10.)
        result={"rgb":image,"nir":image*.8,"path":f"{self.partition}_{index}"}
        if self.labelled or self.poison:result["label"]=index%3
        return result


class FeatureCacheChecks(unittest.TestCase):
    def test_selected_feature_cache_contains_no_target_training_labels_or_model_parameters(self):
        model=loss_checks.CrossmodalChecks().make_model("mcc")
        old={k:v.clone() for k,v in model.state_dict().items()}
        views=[View(p) for p in ("source_train","target_train","target_val")]
        with tempfile.TemporaryDirectory() as temporary:
            record=cache_selected_features(model,views,"cpu",temporary,7,batch_size=2)
            with np.load(Path(temporary)/"selected_feature_cache.npz",allow_pickle=False) as bank:
                self.assertNotIn("target_train_labels",bank.files)
                self.assertEqual(bank["source"].shape,(6,1024))
                self.assertEqual(bank["target_val_labels"].tolist(),[0,1,2,0,1,2])
                self.assertEqual(bank["target_train_paths"].tolist(),[f"target_train_{i}" for i in range(6)])
            self.assertEqual(record["epoch"],7)
            self.assertFalse(record["statistics_or_models_fitted"])
            self.assertEqual(list(Path(temporary).glob("*.pth")),[])
        for k,v in old.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)

    def test_poisoned_target_labels_rejected(self):
        model=loss_checks.CrossmodalChecks().make_model("mcc")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                cache_selected_features(model,[View("source_train"),View("target_train",True),View("target_val")],"cpu",temporary,1)
            self.assertFalse((Path(temporary)/"selected_feature_cache.npz").exists())

    def test_ir_inversion_retains_vis_labels_geometry_and_default_rng(self):
        infrared=torch.linspace(-1,1,2*3*8*8).reshape(2,3,8,8)
        batch={"rgb":torch.rand_like(infrared),"nir":infrared,"label":torch.tensor([0,1])}
        rng=torch.get_rng_state()
        unchanged,_=augment_source_ir_polarity(batch,0.)
        self.assertIs(unchanged,batch);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        changed,record=augment_source_ir_polarity(batch,1.,torch.Generator().manual_seed(42))
        torch.testing.assert_close(changed["rgb"],batch["rgb"],rtol=0,atol=0)
        torch.testing.assert_close(changed["label"],batch["label"],rtol=0,atol=0)
        torch.testing.assert_close(changed["nir"],-infrared,rtol=0,atol=0)
        torch.testing.assert_close((changed["nir"]+1)/2,1-(infrared+1)/2)
        self.assertEqual(record["source_ir_inverted"],2)
        self.assertFalse(record["target_ir_inverted"])

    def test_attention_diagnostic_rows_follow_the_feature_paths_without_target_labels(self):
        model=loss_checks.CrossmodalChecks().make_model('mcc')
        original=model.nir_encoder.forward
        def forward(image):
            value=original(image);first=image.mean((1,2,3)).sigmoid()
            model.nir_encoder.last_pooling_attention=torch.stack((first,1-first),1)
            return value
        views=[View(p) for p in ('source_train','target_train','target_val')]
        with tempfile.TemporaryDirectory() as name,patch.object(model.nir_encoder,'forward',side_effect=forward):
            record=cache_selected_features(model,views,'cpu',name,1,batch_size=2)
            with np.load(Path(name)/'selected_feature_cache.npz',allow_pickle=False) as bank:
                expected=torch.sigmoid(torch.arange(6)*.08).numpy()
                np.testing.assert_allclose(bank['target_train_IR_attention'][:,0],expected,atol=1e-6)
                self.assertNotIn('target_train_labels',bank.files)
                self.assertEqual(bank['target_train_paths'].tolist(),[f'target_train_{i}' for i in range(6)])
            self.assertEqual(record['infrared_attention_maps']['target_train']['samples'],6)


if __name__=="__main__":unittest.main()
