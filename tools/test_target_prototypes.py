import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from weather.target_prototypes import PosteriorTargetPrototypes
from tools.probe_posterior_target_prototypes import load_aligned_probabilities


class TargetPrototypeChecks(unittest.TestCase):
    def test_posterior_columns_preserve_class_identity(self):
        x=np.array([[1.,0],[.95,.05],[0.,1],[.05,.95]])
        p=np.array([[.99,.01],[.9,.1],[.01,.99],[.1,.9]])
        model=PosteriorTargetPrototypes().fit(x,p)
        self.assertEqual(model.predict(x).tolist(),[0,0,1,1])
        swapped=PosteriorTargetPrototypes().fit(x,p[:,::-1])
        self.assertEqual(swapped.predict(x).tolist(),[1,1,0,0])
        centers=model.centers.copy();model.predict(np.array([[-100.,100.]]))
        np.testing.assert_equal(model.centers,centers)

    def test_missing_posterior_classes_are_not_mapped_from_validation(self):
        x=np.eye(3);p=np.tile([.9,.05,.05],(3,1))
        model=PosteriorTargetPrototypes().fit(x,p)
        self.assertEqual(model.active_classes.tolist(),[0])
        self.assertEqual(model.predict(x).tolist(),[0,0,0])

    def test_probability_records_align_by_path_and_reject_target_truth(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);path=run/'predictions.jsonl'
            rows=[dict(epoch=2,partition='target_train',sample_id=k,labels_used=False,probabilities=p)
                  for k,p in [('b',[.2,.8]),('a',[.7,.3])]]
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            np.testing.assert_allclose(load_aligned_probabilities(run,2,np.array(['a','b'])),[[.7,.3],[.2,.8]])
            rows[0]['y_true']=1;path.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(RuntimeError):load_aligned_probabilities(run,2,np.array(['a','b']))


if __name__=='__main__':unittest.main()
