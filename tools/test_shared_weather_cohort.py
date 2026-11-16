import unittest
from tools.prepare_paired_protocol_lists import shared_training_cohort


class SharedWeatherCohortChecks(unittest.TestCase):
    def test_intersection_excludes_both_nonshared_class_sets_and_retains_all_eligible_samples(self):
        source=[('s0',0),('s2a',2),('s2b',2),('s4',4)]
        target=[('t2a',2),('t2b',2),('t5',5)]
        validation=[('v2',2),('v5',5)]
        train,val,audit=shared_training_cohort(source,target,validation)
        self.assertEqual(train,[('t2a',2),('t2b',2)])
        self.assertEqual(val,[('v2',2)])
        self.assertEqual(audit['shared_original_support'],[2])
        self.assertEqual(audit['source_classes_excluded'],[0,4])
        self.assertEqual(audit['target_classes_excluded'],[5])
        self.assertEqual(audit['selected_source_train_samples'],2)

    def test_source_superset_keeps_target_sample_order_and_cohort_unchanged(self):
        source=[('s0',0),('s2',2),('s5',5)]
        target=[('t2',2),('t0',0)];validation=[('v0',0),('v2',2)]
        train,val,audit=shared_training_cohort(source,target,validation)
        self.assertEqual(train,target);self.assertEqual(val,validation)
        self.assertEqual(audit['target_train_samples_excluded'],0)

    def test_disjoint_training_sets_fail_before_lists_or_batches_are_created(self):
        with self.assertRaisesRegex(ValueError,'no shared class'):
            shared_training_cohort([('s0',0)],[('t2',2)],[('v2',2)])


if __name__=='__main__':unittest.main()
