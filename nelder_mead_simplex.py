# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 09:51:36 2022

@author: arbra
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    '''Rosenbrock function for testing, input is arry type (x,y)'''
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2


alpha, beta, gamma, delta = 1, 2, 0.5, 0.5

guess = np.array([-1, 1])
bounds = np.array([[-1, 1], 
                   [-1, 1]])


class Simplex:
    def __init__(self, verts, factors):
        self.verts = verts
        self.dims = verts.shape[0] - 1
        self.sorted = False
        self.values = None
        self.alpha, self.gamma, self.beta, self.delta = factors
        self.xb, self.xw, self.xl, self.xa = None, None, None, None
    
    def evaluate(self, inputs, f):
        res = np.zeros((inputs.shape[0], 1))
        for i, row in enumerate(inputs):
            val = f(row)
            res[i] = val
        return res
    
    def sort(self, f):
        if self.values is None: 
            self.values = self.evaluate(self.verts, f)
            
        sorted_idx = np.argsort(self.values.ravel())
        self.values = self.values[sorted_idx]
        self.verts = self.verts[sorted_idx]
        self.xb = self.verts[0]
        self.xw = self.verts[-1]
        self.xl = self.verts[-2]
        self.sorted = True
    
    def centroid(self):
        if not self.sorted: self.sort()
        self.xa = np.sum(self.verts[:-1], axis=0) / self.dims
        
    def reflection(self):
        self.xr = self.xa + self.alpha * (self.xa - self.xw)
    
    def expansion(self):
        self.xe = self.xr + self.gamma * (self.xr - self.xa)
    
    def contraction(self, direction):
        if direction == 'inside':
            self.xc = self.xa - self.beta * (self.xa - self.xw)
        else:
            self.xo = self.xa + self.beta * (self.xa - self.xw)
    
    def shrinking(self):
        shrunk = self.verts.copy()
        for i in range(1, shrunk.shape[0]):
            shrunk[i] = self.xb + self.delta * (shrunk[i] - self.xb)
        return shrunk


class NelderMeadSimplex:
    def __init__(self, alpha, gamma, beta, delta, guess, bounds, function):
        self.factors = [alpha, beta, gamma, delta]
        self.x0 = guess
        self.bounds = bounds
        self.dims = self.bounds.shape[0]
        self.side_length = 1
        self.f = function
        self.simplicies = []
        
    def initial_simplex(self, x0):
        n = self.dims
        c = self.side_length
        b = (c / (n * np.sqrt(2))) * (np.sqrt(n+1) - 1)
        a = b + c / np.sqrt(2)
        
        v0 = np.zeros((1, n))[0]
        v0.fill(b)
        v0[0] = a
        xx = np.zeros((n+1, n))
        xx[-1] = x0
        
        for i in range(self.dims):
            xx[i] = v0
            v0 = np.roll(v0, 1)
        
        print(a, b, '\n', xx)
        print(np.linalg.norm(xx[0] - xx[1]))
        print(np.linalg.norm(xx[1] - xx[2]))
        print(np.linalg.norm(xx[0] - xx[2]))
        return xx
    
    def opt(self):
        verts = self.initial_simplex(self.x0)
        err = np.inf
        itr = 0
        while itr < 100 or err < 1e-5:
            s = Simplex(verts, self.factors)
            self.simplicies.append(s)
            s.sort(self.f)
            s.centroid()
            s.reflection()
            
            verts = s.verts.copy()
            fr = s.evaluate(np.array([s.xr]), self.f)
            fb = s.values[0]
            fw = s.values[-1]
            fl = s.values[-2]
            if fr < fb:
                s.expansion()
                fe = s.evaluate(np.array([s.xe]), self.f)
                verts[-1] = s.xe if fe < fb else s.xr         
            elif fr <= fl:
                verts[-1] = s.xr
            else:
                if fr > fw:
                    s.contraction('inside')
                    fc = s.evaluate(np.array([s.xc]), self.f)
                    if fc < fw:
                        verts[-1] = s.xc
                    else:
                        verts = s.shrinking()
                else:
                    s.contraction('outside')
                    fc = s.evaluate(np.array([s.xo]), self.f)
                    if fc <= fr:  
                        verts[-1] = s.xo
                    else:
                        verts = s.shrinking()               
            itr += 1
            # break
        
        print(s.verts, s.values)


NM = NelderMeadSimplex(alpha, gamma, beta, delta, guess, bounds, f)
NM.opt()

for s in NM.simplicies:
    tri = s.verts.tolist()
    tri.append(tri[0])
    tri = np.array(tri)
# tri = NM.init_simplex.tolist()
# tri.append(tri[0])
# tri = np.array(tri)
# print(tri)

    plt.plot(tri[:, 0], tri[:, 1])
plt.xlim(-1.5, 1.5)
plt.ylim(-1, 2)
plt.show()
