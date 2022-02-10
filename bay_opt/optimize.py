import os

import numpy as np
import matplotlib.pyplot as plt

from gpr import GaussianProcess
from space import Space
from opt import Optimizer

inputs = {i: [int, [0, 10]] for i in range(10)}

def f(x):
    return np.sum(x)


# initialize surrogate
n_init = 25
n_iters = 50
gp = GaussianProcess()
space = Space(inputs)
opt = Optimizer(space=space,
                gpr=gp,
                n_iters=n_iters,
                n_init=n_init,
                objective=f)

data = np.zeros((n_init + n_iters, 10))


best = np.inf
best_list = []
for i in range(opt.n_init + opt.n_iters):
    print(i)
    next_x = opt.ask()
    print(space.transform(next_x))
    obj_res = opt.objective(space.transform(next_x)[0])
    next_y = obj_res

    space.update_container(i, [*next_x, next_y])
    opt.tell()
    opt.itr += 1
    if best == np.inf or next_y > best:
        best_list.append(next_y)
        best = next_y
    else:
        best_list.append(best)


plt.plot(best_list)
plt.show()
