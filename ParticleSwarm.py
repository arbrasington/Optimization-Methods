# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:09:10 2022

@author: ALEXRB
"""
import abc
from math import floor, ceil
from random import random, uniform
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import os


class Particle:
    """
    Particle class
    """

    def __init__(self, bounds):
        self.dims = len(bounds)

        self.position = [random() for _ in range(self.dims)]
        self.velocity = [uniform(-1, 1) for _ in range(self.dims)]

        self.best_pos = self.position.copy()

        self.err = -1
        self.best_err = -1

    def evaluate(self, costFunc):
        """
        Evaluate current fitness with the given function
        """
        self.err = costFunc(self.position)

        # check to see if the current position is an individual best
        if self.err > self.best_err or self.best_err == -1:
            self.best_pos = self.position.copy()
            self.best_err = self.err

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
    def update_position(self, bounds):
        """
        Update the particle's position based off the velocities
        """
        for i in range(self.dims):
            self.position[i] = self.position[i] + self.velocity[i]

            # adjust maximum position if necessary
            if self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

            # adjust minimum position if necessary
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]


class ParticleSwarmOptimizer(abc.ABC):
    """
    Particle swarm optimizer for maximizing an objective function
    """
    best_pos = []
    best_err = -1

    def __init__(self):
        self.pop_size = 15
        self.generations = 10
        self.particles = self.pop_size
        self.max_iters = self.generations
        self.bounds = []
        self.particle_bounds = []
        self.bestlist = []

        self.stop = False
        self.is_running = False

        self.swarm = [Particle([[0, 1]] * len(self.bounds)) for _ in range(self.particles)]  # initialize swarm
        self.results = np.zeros((len(self.swarm) * self.max_iters, len(self.bounds) + 2))  # results container

    def run(self):
        """
        Run the optimization
        """
        self.is_running = True
        self.running.signal.emit()
        self.particle_bounds = [[0, 1]] * len(self.bounds)
        self.particles = self.pop_size
        self.max_iters = self.generations
        self.swarm = [Particle([[0, 1]] * len(self.bounds)) for _ in range(self.particles)]
        self.results = np.zeros((len(self.swarm) * self.max_iters, len(self.bounds) + 2))

        for i in range(self.max_iters):

            if self.stop:
                self.is_running = False
                return

            # cycle through particles
            for j in range(len(self.swarm)):
                self.swarm[j].evaluate(self.fitness)
                self.update_results(self.swarm[j], j + self.particles * i)

                # determine if this particle is the best
                if self.swarm[j].err > self.best_err or self.best_err == -1:
                    self.best_pos = self.swarm[j].position.copy()
                    self.best_err = deepcopy(self.swarm[j].err)
                    self.bestlist.append(self.best_err)

            # update velocities and positions
            for j in range(len(self.swarm)):
                self.swarm[j].update_velocity(self.best_pos)
                self.swarm[j].update_position(self.particle_bounds)

            if self.convergence(i): return

    def stop_run(self):
        self.stop = True
        while self.is_running:
            pass
        self.stopped.signal.emit()

    def reinitialize(self):
        """
        Reset all variables and create a new swarm
        """
        self.swarm = [Particle([[0, 1]] * len(self.bounds)) for _ in range(self.particles)]  # initialize swarm
        self.results = np.zeros((len(self.swarm) * self.max_iters, len(self.bounds) + 2))  # results container
        self.best_err, self.best_pos = -1, []
        self.stop = False
        self.reset.signal.emit()

    def update_results(self, particle: Particle, index):
        """
        Update the results array and save to file
        """
        itr = floor(index / self.particles) + 1
        self.results[index] = [itr, *particle.position, particle.err]
        self.updated.signal.emit(self.results[~np.all(self.results == 0, axis=1)])

        # if self.filename is not None and isinstance(self.filename, str):
        #     with open(self.filename, 'a') as f:
        #         if index == 0:
        #             f.truncate(0)  # reset the file
        #         np.savetxt(f, [itr, *particle.position, particle.err], newline=',')
        #         f.write("\n")
        # f.close()

    def convergence(self, iteration, value=1e-10):
        """
        Test for convergence based on variance of current swarm
        """
        data = self.results[self.results[:, 0] == iteration + 1, 1:-1]
        return np.all(np.var(data, axis=0) <= value)

    @abc.abstractmethod
    def fitness(self, inputs):
        pass

    @staticmethod
    def continuous_to_int(continuous, lower_bound, upper_bound):
        """
        Convert the continuous variable to its corresponding integer value
        """
        integer = floor((upper_bound - lower_bound + 1) * continuous) + lower_bound
        if integer > upper_bound: integer = upper_bound
        if integer < lower_bound: integer = lower_bound
        return integer

    @staticmethod
    def continuous_to_real(continuous, lower_bound, upper_bound):
        """
        Convert the continuous variable to its corresponding real value
        """
        return (upper_bound - lower_bound) * continuous + lower_bound


class Test_PSO(ParticleSwarmOptimizer):
    def __init__(self, bounds, particles, iterations, filename):
        super().__init__(bounds, particles, iterations, filename)

    def fitness(self, inputs):
        return sum([i ** 2 for i in inputs])


if __name__ == '__main__':
    bounds = [(0, 1), (0, 1), (0, 6), (8, 16)]

    pso = Test_PSO(bounds=bounds,
                   particles=20,
                   iterations=10,
                   filename='test.csv')
    pso.run()