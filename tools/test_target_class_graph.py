import unittest
import numpy as np
from scipy.optimize import linear_sum_assignment
from tools.probe_target_class_graph import match_graphs


class ClassGraphChecks(unittest.TestCase):
    def test_relative_geometry_recovers_unknown_node_order_without_cross_domain_coordinates(self):
        a=np.array([[0,1,4,9],[1,0,2,8],[4,2,0,3],[9,8,3,0]],dtype=float)
        target_order=np.array([2,0,3,1]);b=a[target_order][:,target_order]*4
        fit=match_graphs(a,b,np.ones_like(a),gammas=(0.,))[0]
        self.assertEqual(fit['cluster_to_source_class'],target_order.tolist())
        self.assertLess(fit['fit_objective'],1e-10)

    def test_pure_cross_cost_matches_hungarian_and_assignment_is_always_bijective(self):
        a=np.ones((3,3))-np.eye(3);cost=np.array([[4,1,5],[6,5,0],[0,6,5]],dtype=float)
        fit=match_graphs(a,a,cost,gammas=(0.,.5,1.))
        rows,columns=linear_sum_assignment(cost);mapping=np.empty(3,dtype=int);mapping[columns]=rows
        self.assertEqual(fit[-1]['cluster_to_source_class'],mapping.tolist())
        for row in fit:self.assertEqual(set(row['cluster_to_source_class']),{0,1,2})


if __name__=='__main__':unittest.main()
