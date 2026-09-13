"""Numerical checks of the paper's distribution and calibration identities."""
import unittest
import numpy as np
import torch
from scipy.stats import nbinom
from src.loss.ztnb import ztnb_nll, compute_conditional_mean
from src.calibration.bin_calibration import calibrate_kbins
from src.data.yd_extractor import extract_yd_kbins
from src.training.evaluate import compute_cpc_pair


class PaperFormulaTests(unittest.TestCase):
    def test_ztnb_normalization_mean_and_loss(self):
        for mu, phi in [(0.1, 0.7), (3., 2.), (20., 5.)]:
            t = np.arange(1, 2000)
            p0 = (phi / (phi + mu)) ** phi
            pmf = nbinom.pmf(t, phi, phi / (phi + mu)) / (1-p0)
            self.assertAlmostEqual(float(pmf.sum()), 1., places=10)
            self.assertAlmostEqual(float((t*pmf).sum()), mu/(1-p0), places=9)
            obs=torch.tensor([1.,3.,8.],dtype=torch.float64)
            log_phi=torch.tensor(np.log(phi),dtype=torch.float64)
            means=torch.full_like(obs,mu)
            expected=-np.log(nbinom.pmf(obs.numpy(),phi,phi/(phi+mu))/(1-p0)).mean()
            self.assertAlmostEqual(float(ztnb_nll(obs,means,log_phi)),expected,places=5)
            np.testing.assert_allclose(compute_conditional_mean(means,log_phi).numpy(),mu/(1-p0),rtol=1e-6)

    def test_empty_bins_and_fractional_q(self):
        distance=np.array([1.,2.,11.,12.]);edges=np.array([0.,5.,10.,np.inf])
        pred=np.array([2.,4.,3.,6.]);mask=np.ones(4,dtype=bool);target=np.array([.7,0.,.3])
        for q in [0.,.2,.5,1.]:
            result=calibrate_kbins(pred,distance,mask,target,edges,q=q)
            self.assertAlmostEqual(result.sum(),pred.sum())
            self.assertAlmostEqual(result[0]/result[1],pred[0]/pred[1])
            self.assertAlmostEqual(result[2]/result[3],pred[2]/pred[3])
            if q==0:np.testing.assert_allclose(result,pred)
            if q==1:np.testing.assert_allclose(extract_yd_kbins(distance,result,edges,mask),target)

    def test_oracle_matching_does_not_guarantee_cpc_gain(self):
        truth=np.array([9.,1.,1.,9.]);pred=np.array([9.,.1,1.9,9.])
        distance=np.array([1.,2.,11.,12.]);edges=np.array([0.,5.,np.inf]);mask=np.ones(4,dtype=bool)
        target=extract_yd_kbins(distance,truth,edges,mask)
        result=calibrate_kbins(pred,distance,mask,target,edges)
        np.testing.assert_allclose(extract_yd_kbins(distance,result,edges,mask),target)
        self.assertLess(compute_cpc_pair(truth,result),compute_cpc_pair(truth,pred))
        self.assertAlmostEqual(compute_cpc_pair(truth,result),1-np.abs(truth-result).sum()/(truth.sum()+result.sum()))

    def test_zero_target_breaks_positive_support_for_positive_q(self):
        pred=np.array([2.,3.]);dist=np.array([1.,11.]);edges=np.array([0.,5.,np.inf]);mask=np.ones(2,dtype=bool)
        result=calibrate_kbins(pred,dist,mask,np.array([0.,1.]),edges,q=.5)
        self.assertEqual(result[0],0.)
        self.assertAlmostEqual(result.sum(),pred.sum())
