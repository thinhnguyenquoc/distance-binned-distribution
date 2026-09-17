import unittest
import numpy as np
from src.experiment.run_unified_placebo import stratified_indices, bootstrap_ci

class PlaceboBootstrapTests(unittest.TestCase):
    def test_fold_counts_preserved_even_with_unequal_sizes(self):
        folds=np.array([1,1,2,2,2])
        indices=stratified_indices(folds,200,42)
        self.assertTrue(((folds[indices]==1).sum(axis=1)==2).all())
        self.assertTrue(((folds[indices]==2).sum(axis=1)==3).all())
        np.testing.assert_array_equal(indices,stratified_indices(folds,200,42))

    def test_stratification_excludes_fold_weight_variation(self):
        indices=stratified_indices(np.repeat([1,2],10))
        values=np.repeat([0.,10.],10)
        self.assertEqual(bootstrap_ci(values,indices),(5.,5.))

    def test_paired_constant_difference_stays_constant(self):
        indices=stratified_indices(np.repeat([1,2],10))
        target=np.arange(20,dtype=float)
        placebo=target-2
        self.assertEqual(bootstrap_ci(target-placebo,indices),(2.,2.))

    def test_reject_nan(self):
        with self.assertRaises(ValueError):
            bootstrap_ci(np.array([np.nan,1]),stratified_indices(np.array([1,1]),10))
