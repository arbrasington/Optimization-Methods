from time import time
from typing import Callable

from sampling import HaltonSequenceSampler
from space import Space

from acq import propose_location, expected_improvement
from gpr import GaussianProcess


class Optimizer:
    space: Space = None
    gpr: GaussianProcess = None
    n_iters: int = None
    n_init: int = None
    objective: Callable = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                self.__dict__.update([(key, value)])
            except:
                print('Invalid key')

        self.itr = 0
        self.space.init_container(self.n_init + self.n_iters)
        self.init_points = HaltonSequenceSampler(self.space.dims, self.n_init)

    def optimize(self):
        if self.n_init <= 1: return
        for i in range(self.n_init + self.n_iters):
            print(f'Iteration: {i}')
            next_x = self.ask()
            y = self.objective(self.space.transform(next_x))
            self.space.update_container(i, [*next_x, y])
            self.tell()
            self.itr += 1

    def ask(self):
        tic = time()
        if self.itr < self.n_init:
            loc = self.init_points[self.itr]

        else:
            data = self.space.get_nonzero()
            loc = propose_location(expected_improvement,
                                   data[:, :-1], data[:, -1],
                                   self.gpr.model,
                                   self.space.bounds)

        return loc

    def tell(self):
        data = self.space.get_nonzero()
        self.gpr.set_training_data(data[:, :-1], data[:, -1])
        tic = time()
        self.gpr.train()
