from __future__ import annotations

from typing import List, Tuple

import mitsuba as mi

from emitters.util import kernel_gather, kernel_linear_search


class BatchDiscreteDistribution:
    def __init__(self, pmf: List[mi.Float]):
        super().__init__()
        self.m_pmf: List[mi.Float] = pmf
        self.m_cdf: List[mi.Float] = []
        self.m_sum: mi.Float = 0.
        self.m_normalization: mi.Float = 1.
        self.compute_cdf(self.m_pmf)

    def pmf(self):
        return self.m_pmf

    def cdf(self):
        return self.m_cdf

    def sum(self):
        return self.m_sum

    def normalization(self):
        return self.m_normalization

    def size(self):
        return len(self.m_pmf)

    def empty(self):
        return len(self.m_pmf) == 0

    def eval_pmf(self, index: mi.Int | int, active: bool = True):
        return kernel_gather(self.m_pmf, index, active)

    def eval_pmf_normalized(self, index: mi.Int | int, active: bool = True):
        return kernel_gather(self.m_pmf, index, active) * self.m_normalization

    def eval_cdf(self, index: mi.Int | int, active: bool = True):
        return kernel_gather(self.m_cdf, index, active)

    def eval_cdf_normalized(self, index: mi.Int | int, active: bool = True) -> mi.Float:
        return kernel_gather(self.m_cdf, index, active) * self.m_normalization

    def sample(self, value: mi.Float, active: bool = True) -> mi.Int:
        value = value * self.m_sum
        return kernel_linear_search(self.m_cdf, value, active)

    def sample_pmf(self, value: mi.Float, active: bool = True) -> Tuple[mi.Int, mi.Float]:
        index = self.sample(value, active)
        return index, self.eval_pmf_normalized(index)

    def sample_reuse(self, value: mi.Float, active: bool = True) -> Tuple[mi.Int, mi.Float]:
        index = self.sample(value, active)
        pmf = self.eval_pmf_normalized(index, active)
        cdf = self.eval_cdf_normalized(index - 1, active & (index > 0))
        return index, (value - cdf) / pmf

    def sample_reuse_pmf(self, value: mi.Float, active: bool = True):
        index, pdf = self.sample_pmf(value, active)
        pmf = self.eval_pmf_normalized(index, active)
        cdf = self.eval_cdf_normalized(index - 1, active & (index > 0))
        return index, (value - cdf) / pmf, pmf

    def compute_cdf(self, pmf):
        if len(pmf) == 0:
            raise ValueError('DiscreteDistribution: empty distribution!')

        cdf = []
        sum = mi.Float(0.)
        for i in range(len(pmf)):
            sum = sum + pmf[i]
            cdf.append(sum)
        self.m_sum = sum
        self.m_normalization = 1.0 / sum
        self.m_cdf = cdf
