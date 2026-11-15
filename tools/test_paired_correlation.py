import copy
import unittest
import numpy as np
from weather.paired_correlation import PairedCorrelation


class PairedCorrelationChecks(unittest.TestCase):
    def arrays(self):
        rng=np.random.default_rng(42)
        latent=rng.standard_normal((800,2));a=rng.standard_normal((2,5));b=rng.standard_normal((2,7))
        vis=latent@a+.01*rng.standard_normal((800,5))
        ir=latent@b+.01*rng.standard_normal((800,7))
        labels=(latent[:400,0]>0).astype(np.int64)
        return vis[:400]+10,ir[:400]-4,vis[400:]-5,ir[400:]+8,labels

    def test_finds_shared_coordinates_with_unequal_widths_and_separate_domain_offsets(self):
        arrays=self.arrays();cca=PairedCorrelation.fit(*arrays,regularization=.001)
        self.assertTrue(np.all(cca.correlations[:2]>.99))
        z=cca.transform(arrays[2],arrays[3],"target",2)
        self.assertEqual(z.shape,(400,4))
        np.testing.assert_allclose(np.linalg.norm(z,axis=1),1)

    def test_pair_identity_matters(self):
        a=list(self.arrays());aligned=PairedCorrelation.fit(*a,regularization=.001,target_mixture=1)
        a[3]=a[3][np.random.default_rng(91).permutation(400)]
        shuffled=PairedCorrelation.fit(*a,regularization=.001,target_mixture=1)
        self.assertGreater(aligned.correlations[:2].mean(),shuffled.correlations[:2].mean()+.5)

    def test_source_balancing_and_transform_do_not_refit_on_validation(self):
        arrays=self.arrays();cca=PairedCorrelation.fit(*arrays,source_covariance="class_between")
        before=copy.deepcopy(cca.__dict__)
        cca.transform(np.full((3,5),1e6),np.full((3,7),-1e6),"target",2)
        self.assertEqual(cca.metadata,before["metadata"])
        for x,y in zip(cca.projections,before["projections"]):np.testing.assert_array_equal(x,y)
        self.assertFalse(cca.metadata["target_training_labels_used"])
        self.assertFalse(cca.metadata["validation_used_for_fit"])
        with self.assertRaises(TypeError):PairedCorrelation.fit(*arrays,target_labels=np.zeros(400))


if __name__=="__main__":unittest.main()
