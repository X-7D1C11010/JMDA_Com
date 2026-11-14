import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from tools.probe_weather_feature_geometry import TrainingGeometry, source_cluster_mapping, prototype_cluster_mapping


class FeatureGeometryTests(unittest.TestCase):
    def test_train_only_domain_center_corrects_translation_without_fitting_validation(self):
        generator = np.random.default_rng(42)
        source = np.r_[generator.normal((-2., 0.), .1, (20, 2)), generator.normal((2., 0.), .1, (20, 2))]
        labels = np.repeat((0, 1), 20)
        target = source+np.array((8., 3.))
        geometry = TrainingGeometry("domain_center", 2).fit(source, target)
        before = geometry.target_mean.copy()
        a = geometry.transform(source, "source")
        b = geometry.transform(target, "target")
        np.testing.assert_allclose(a, b, atol=1e-12)
        classifier = LogisticRegression().fit(a, labels)
        np.testing.assert_array_equal(classifier.predict(b), labels)
        geometry.transform(np.full((5, 2), 10000.), "target")
        np.testing.assert_array_equal(before, geometry.target_mean)
        self.assertEqual(geometry.fit_samples, {"source_train": 40, "target_train": 40})

    def test_cluster_mapping_is_derived_from_source_semantics(self):
        mapping = source_cluster_mapping(np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]), 2)
        np.testing.assert_array_equal(mapping, [1, 0])

    def test_prototype_association_recovers_class_permutation_without_target_labels(self):
        source = np.array([[1., 0.], [1., .1], [0., 1.], [.1, 1.]])
        mapping = prototype_cluster_mapping(source, np.array([0, 0, 1, 1]), np.array([[0., 1.], [1., 0.]]), 2)
        np.testing.assert_array_equal(mapping, [1, 0])


if __name__ == "__main__":
    unittest.main()
