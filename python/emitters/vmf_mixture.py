from typing import List

import drjit as dr
import mitsuba as mi

from emitters.distr_1d import BatchDiscreteDistribution
from emitters.util import trunc_exp, VonMisesFisher, kernel_gather_vmf, sigmoid


class VonMisesFisherMixture:
    def __init__(self, feature: List[mi.Float], coord_dim=2):
        assert coord_dim in [2, 3]
        self.n_lobe = len(feature) // (2 + coord_dim)
        self.vMFs = []
        probs = []
        for i in range(self.n_lobe):
            if coord_dim == 2:
                u = sigmoid(feature[i])
                v = sigmoid(feature[self.n_lobe + i])
                kappa = trunc_exp(feature[self.n_lobe * 2 + i])
                prob = trunc_exp(feature[self.n_lobe * 3 + i])
                self.vMFs.append(VonMisesFisher.from_theta_phi(u, v, kappa))
            else:
                x = feature[i]
                y = feature[self.n_lobe + i]
                z = feature[self.n_lobe * 2 + i]
                kappa = trunc_exp(feature[self.n_lobe * 3 + i])
                prob = trunc_exp(feature[self.n_lobe * 4 + i])
                self.vMFs.append(VonMisesFisher(dr.normalize(mi.Vector3f(x, y, z)), kappa))
            probs.append(prob)
        self.weight_warp = BatchDiscreteDistribution(probs)

    def sample_direction(self, sample: mi.Point2f, active: bool = True):
        index, sample_x_re = self.weight_warp.sample_reuse(sample.x, active=active)
        sample.x = sample_x_re
        sel_vmf = kernel_gather_vmf(self.vMFs, index, active)
        d = sel_vmf.sample_direction(sample)
        return d

    def pdf_direction(self, d: mi.Vector3f, active: bool = True):
        pdf_sum = mi.Float(0.)
        for i in range(self.n_lobe):
            pdf_sum += self.vMFs[i].pdf_direction(d) * self.weight_warp.eval_pmf_normalized(i, active)
        return pdf_sum
