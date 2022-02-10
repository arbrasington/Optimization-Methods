# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:09:10 2022

@author: ALEXRB
"""
import abc
from random import random, uniform
from copy import deepcopy
from typing import Callable

import numpy as np

from transformer import transform


class Particle:
    """
    Particle class
    """

    def __init__(self, dims, max_min):
        self.dims = dims
        self.max_min = max_min

        self.position = np.random.rand(self.dims)
        self.velocity = np.random.uniform(-1., 1., self.position.shape)

        self.best_pos = self.position.copy()

        self.err = -np.inf if self.max_min == 'max' else np.inf
        self.best_err = -np.inf if self.max_min == 'max' else np.inf

    def evaluate(self, costFunc, var_types, bounds):
        """
        Evaluate current fitness with the given function
        """
        self.err = costFunc(self.decode(var_types, bounds))
        
        if self.max_min == 'max':
            if self.err > self.best_err:
                self.best_pos = self.position.copy()
                self.best_err = self.err
        else:
            if self.err < self.best_err:
                self.best_pos = self.position.copy()
                self.best_err = self.err
    
    def decode(self, types, bounds):
        decoded = np.zeros(self.position.shape)
        for i in range(decoded.shape[0]):
            decoded[i] = transform(self.position[i],
                                   types[i],
                                   bounds[i])
        return decoded

    def update_velocity(self, pos_best_g):
        """
        Update particle velocity
        """
        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognitive constant
        c2 = 2  # social constant

        for i in range(self.dims):
            vel_cognitive = c1 * random() * (self.best_pos[i] - self.position[i])
            vel_social = c2 * random() * (pos_best_g[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
        """
        Update the particle's position based off the velocities
        """
        maxx = 1
        minn = 0

        for i in range(self.dims):
            self.position[i] = self.position[i] + self.velocity[i]

            # adjust maximum position if necessary
            if self.position[i] > maxx:
                self.position[i] = maxx

            # adjust minimum position if necessary
            if self.position[i] < minn:
                self.position[i] = minn


class ParticleSwarmOptimizer:
    """
    Particle swarm optimizer for maximizing an objective function
    """
    best_pos = []
    best_err = -1

    def __init__(self, input_dict: dict,
                 particles: int,
                 max_itrs: int,
                 fitness: Callable,
                 opt_type='max'):

        self.var_types = np.asarray([value[0] for _, value in input_dict.items()])
        self.bounds = np.asarray([value[1] for _, value in input_dict.items()])

        self.dims = self.bounds.shape[0]
        self.particles = particles
        self.max_iters = max_itrs
        self.fitness = fitness
        self.opt_type = opt_type

        self.swarm = [Particle(self.dims, self.opt_type) for _ in range(self.particles)]  # initialize swarm

    def run(self):
        """
        Run the optimization
        """
        self.best_err = -np.inf if self.opt_type == 'max' else np.inf

        best_list = []
        for i in range(self.max_iters):
            print(f'Iteration {i+1} of {self.max_iters}')

            # cycle through particles
            for j in range(len(self.swarm)):
                self.swarm[j].evaluate(self.fitness, self.var_types, self.bounds)

                # determine if this particle is the best
                if self.opt_type == 'max':
                    if self.swarm[j].err > self.best_err:
                        self.best_pos = self.swarm[j].position.copy()
                        self.best_err = deepcopy(self.swarm[j].err)
                        self.best_part = self.swarm[j]
                else:
                    if self.swarm[j].err < self.best_err:
                        self.best_pos = self.swarm[j].position.copy()
                        self.best_err = deepcopy(self.swarm[j].err)
                        self.best_part = self.swarm[j]

            # update velocities and positions
            for j in range(len(self.swarm)):
                self.swarm[j].update_velocity(self.best_pos)
                self.swarm[j].update_position()
            
            print(f'Best: {self.best_part.decode(self.var_types, self.bounds)} -> {self.best_part.err}\n')
            best_list.append(self.best_err)

        return best_list


inputs = {i: [int, [0, 10]] for i in range(20)}


def f(x):
    return np.sum(x)


PSO = ParticleSwarmOptimizer(input_dict=inputs,
                            particles=50,
                            max_itrs=50,
                            fitness=f,
                            opt_type='max')
vals = PSO.run()

import matplotlib.pyplot as plt
plt.plot(vals)
plt.show()