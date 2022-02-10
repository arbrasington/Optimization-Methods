import numpy as np
from scipy.stats import qmc


def LatinHyperCubeSampler(dims, n):
    sampler = qmc.LatinHypercube(d=dims)
    return sampler.random(n)


def HaltonSequenceSampler(dims, n):
    sampler = qmc.Halton(d=dims, scramble=True)
    return sampler.random(n)


def UniformSampler(dims, n):
    return np.random.rand(n, dims)
