import unittest
import numpy as np
import torch
from weather.histogram_style import HistogramSourceStyle,histogram_lookup,fit_training_style,cohort_signature


class StyleChecks(unittest.TestCase):
    def stats(self):
        return {'fit_partitions':['source_train','target_train'],'target_instance_labels_used':False,
                'validation_used_for_statistics':False,'lookups':{'rgb':[[.3]*256]*3,'nir':[[.3]*256]}}

    def test_lookup_is_monotone_and_present_bins_identity(self):
        h=np.zeros((1,256));h[0,[10,50,200]]=[2,3,5]
        identity=histogram_lookup(h,h)
        np.testing.assert_allclose(identity[0,[10,50,200]],np.array([10,50,200])/255,atol=1e-7)
        self.assertTrue((np.diff(identity,axis=1)>=0).all())
        with self.assertRaisesRegex(ValueError,'training'):
            fit_training_style([('a','b')],[('c','d')],target_partition='target_val')

    def test_disabled_is_exact_and_strength_retains_intensity_order(self):
        x=torch.zeros(2,3,16,16);x[1]=1
        mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
        b={'rgb':(x-mean)/std,'nir':x*2-1,'label':torch.tensor([0,1])}
        op=HistogramSourceStyle(self.stats());state=torch.get_rng_state().clone()
        out,_=op(b,0.)
        self.assertIs(out,b);self.assertTrue(torch.equal(state,torch.get_rng_state()))
        out,record=op(b,1.,torch.Generator().manual_seed(42),strength=.85)
        self.assertIs(out['label'],b['label'])
        rgb=out['rgb']*std+mean;ir=out['nir']*.5+.5
        self.assertTrue(torch.all(rgb[1]-rgb[0]>.1499));self.assertTrue(torch.all(ir[1]-ir[0]>.1499))
        self.assertFalse(record['target_histogram_augmented']);self.assertEqual(record['source_histogram_augmented'],2)

    def test_cohort_signature_is_portable_but_detects_a_different_sample(self):
        a=[('D:\\Code\\JMDA-Net\\Data\\VIS\\a.jpg','D:\\Code\\JMDA-Net\\Data\\IR\\a.jpg')]
        b=[('/mnt/data/Data/VIS/a.jpg','/mnt/data/Data/IR/a.jpg')]
        c=[('/mnt/data/Data/VIS/b.jpg','/mnt/data/Data/IR/a.jpg')]
        self.assertEqual(cohort_signature(a),cohort_signature(b));self.assertNotEqual(cohort_signature(a),cohort_signature(c))


if __name__=='__main__':unittest.main()
