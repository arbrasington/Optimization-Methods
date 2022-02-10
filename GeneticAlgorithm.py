# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:55:50 2022

@author: ALEXRB
"""

import numpy as np
from typing import Callable
from random import uniform

from transformer import transform


class Chromosome:
    """
    Class for chromosome data and actions
    """
    def __init__(self, genes, dims):
        self.dims = dims
        if genes == 'random':  # make random chromosome
            self.genes = np.random.rand(self.dims)
        elif genes == 'zero':  # make empty chromosome
            self.genes = np.zeros(self.dims)
        else:
            assert(len(genes) == dims)
            self.genes = genes # make chromosome with specified genes
        
        self.fitness = None
    
    def mate(self, chromosome, alpha, beta):
        """
        Mate the current chromosome instance with another.
        Alpha and beta parameters are summarized in:
            https://apmonitor.com/me575/uploads/Main/optimization_book.pdf

        Parameters
        ----------
        chromosome : Chromosome
            Second chromosome to be mated with
        alpha : float
            Alpha parameter for mutation
        beta : float
            Beta parameter for mutation

        Returns
        -------
        child : Chromosome
            Resulting chromosome from cross over and mutation

        """
        # cross over
        binary_mask = np.random.randint(2, size=self.dims)
        child = Chromosome(genes='zero', dims=self.dims)
        
        for i, binary in enumerate(binary_mask):
            # child get gene from current chromosome one if 0 else gene from other
            child.genes[i] = self.genes[i] if binary == 0 else chromosome.genes[i]
        
        child.mutate(alpha, beta)
        
        return child
    
    def mutate(self, alpha, beta, mut_perc = 0.01):
        """
        Mutate the chromosome based on alpha and beta inputs. Mutation formulas
        are from:
            https://apmonitor.com/me575/uploads/Main/optimization_book.pdf

        Parameters
        ----------
        alpha : float
            Determines favorablility of mutation. If alpha < 1, the mutated
            gene will favor the current value. If alpha = 0, no mutation occurs.
            If alpha = 1, mutation is uniform.
        beta : float
            The mutation parameter. Controls how alpha changes with generations.
            If beta > 1 then alpha = 1 for first generation and decreases to 
            near zero in the final generation.
        mut_perc : float, optional
            Percentage below which a gene is mutated. Value is in the range [0,1].
            A higher value results in more mutation. A value of zero results 
            in no mutation. The default is 0.01.

        Returns
        -------
        None.

        """
        genes_mut = self.genes.copy()  # array of mutated genes
        for i, gene in enumerate(self.genes):
            mut_prob = uniform(0, 1)  # random uniform value between 0 and 1
            if mut_prob < mut_perc:
                # mutate current gene
                r = uniform(0, 1)
                if r <= gene:
                    genes_mut[i] = (r ** alpha) * (gene ** (1 - alpha))
                else:
                    genes_mut[i] = 1 - ((1 - r) ** alpha * (1 - gene) ** (1 - alpha))
        
        self.genes = genes_mut 
 

class GeneticAlgorithm:
    """
    Mixed integer genetic algorithm.
    
    Methdology is an adaptation from Haupt and Parkinson et al.
    Details can be found in: 
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4120263
        https://apmonitor.com/me575/uploads/Main/optimization_book.pdf
    """
    
    def __init__(self,
                 input_dict: dict,
                 max_gens: int,
                 pop_size: int,
                 fitness: Callable,
                 beta=0.25,
                 opt_type='max'):
        
        self.var_types = np.asarray([value[0] for _, value in input_dict.items()])
        self.bounds = np.asarray([value[1] for _, value in input_dict.items()])
        
        self.dims = self.bounds.shape[0]
        self.generations = max_gens
        self.pop_size = pop_size
        self.fitness = fitness
        self.beta = beta
        self.opt_type = opt_type
        
        self.population = [Chromosome('random', self.dims) for _ in range(self.pop_size)]
        
        self.gen = 1

    
    def run(self):
        """
        Run the algorithm for defined number of generations

        Returns
        -------
        None.
        """
        self.evaluate_pop(self.population)
        fitness = np.asarray([c.fitness for c in self.population])
        best_vals = []

        while self.gen <= self.generations:
            print(f'Generation {self.gen} of {self.generations}')
            parents = self.tournament_selection(fitness)
            
            child_pop = []
            gen_alpha = self.alpha_function()
            for idx, row in enumerate(parents):  # mate parents to create next population
                chrom1 = self.population[row[0]]
                chrom2 = self.population[row[1]]
                
                child = chrom1.mate(chrom2, gen_alpha, self.beta)
                child_pop.append(child)
            
            self.evaluate_pop(child_pop)
            child_fitness = np.asarray([c.fitness for c in child_pop])
            
            self.population, fitness, best = self.elitism(self.population,
                                                    fitness,
                                                    child_pop,
                                                    child_fitness)
            best_vals.append(best.fitness)
            print(f'Best: {self.decode(best.genes)} -> {best.fitness}\n')
            self.gen += 1
            
        return best_vals
            
    def tournament_selection(self, fitness):
        """
        Choose parents based on tournament selection
            - random chromosomes selected to be in tournament of size tournament_size
            - best chromosome becomes a parent in each tournament

        Parameters
        ----------
        fitness : np.ndarray
            array of fitness values for the current population

        Returns
        -------
        parents : np.ndarray
            An array of parent indexes from current population of shape (pop_size, 2)
        """
        parents = np.zeros((self.pop_size, 2), dtype=int)
        size = int(self.pop_size / 2)
        
        rnd = np.random.default_rng()  # random generator without duplicates
        for i in range(self.pop_size):
            for j in range(2):  # play twice to get two parents
                players = rnd.choice(self.pop_size, size=size, replace=False) # pick random players
                if self.opt_type == 'max':
                    winner = np.argmax(fitness[players])  # pick player with maximum score
                else:
                    winner = np.argmin(fitness[players]) # pick player with minimum score
                    
                parents[i, j] = players[winner] # add winning chromosome to parents
                
        return parents
    
    def elitism(self, parent_pop, parent_fit, child_pop, child_fit):
        """
        Perform elitism. Forces children to compete against parents to survive 
        to next generation. Best chromosomes move to next population.

        Parameters
        ----------
        parent_pop : list
            Current population
        parent_fit : np.ndarray
            Fitness of current population.
        child_pop : list
            Child population.
        child_fit : np.ndarray
            Fitness of child population.

        Returns
        -------
        np.ndarray
            Array of size (pop_size,) of top chromosomes
        np.ndarray
            Array of size (pop_size,) of top chromosomes' fitness
        np.ndarray
            Current best chromosome
        """
        
        # combine parent and children arrays
        combined_pop = np.concatenate((parent_pop, child_pop))
        combined_fit = np.concatenate((parent_fit, child_fit))
        
        if self.opt_type == 'max':
            best_to_worst = np.flip(np.argsort(combined_fit))  # sort combined fitness high to low
        else:
            best_to_worst = np.argsort(combined_fit)  # sort combined fitness high to low

        # sort population and fitness by high_to_low
        ordered_pop = combined_pop[best_to_worst]
        ordered_fit = combined_fit[best_to_worst]

        # return first n elite designs (n = population size)
        return ordered_pop[:self.pop_size], ordered_fit[:self.pop_size], ordered_pop[0]
    
    def evaluate_pop(self, population):
        """
        Evaluate the given population for fitness values.

        Parameters
        ----------
        population : list
            Population to be evaluated.

        Returns
        -------
        None.

        """
        for chromosome in population:
            chromosome.fitness = self.fitness(self.decode(chromosome.genes))
    
    def decode(self, chromosome):
        """
        Decode the chromosome into applicable inputs to the objective funciton.

        Parameters
        ----------
        chromosome : np.ndarray
            Genes of a chromosome (i.e. Chromosome.genes)

        Returns
        -------
        decoded : np.ndarray
            Decoded chromosomes.

        """
        
        decoded = np.zeros(chromosome.shape)
        for i in range(decoded.shape[0]):
            decoded[i] = transform(chromosome[i], 
                                   self.var_types[i], 
                                   self.bounds[i]) 
        
        return decoded
    
    def alpha_function(self):
        """
        Function to find the alpha value from generations and beta.

        Returns
        -------
        float
            The calculated alpha value.

        """
        return (1 - ((self.gen - 1) / self.generations)) ** self.beta



inputs = {i: [int, [0, 10]] for i in range(10)}

def f(x):
    return np.sum(x)

GA = GeneticAlgorithm(input_dict=inputs,
                      max_gens=50,
                      pop_size=50,
                      fitness=f,
                      beta=0.25,
                      opt_type='min')
vals = GA.run()

import matplotlib.pyplot as plt
plt.plot(vals)
plt.show()
# print(Chromosome('random', 10).genes)
# print(GA.population)

