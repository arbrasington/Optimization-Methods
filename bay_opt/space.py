import numpy as np
from typing import Union, List


def continuous_to_int(continuous: float, lower_bound: int, upper_bound: int) -> int:
    """
    Convert the continuous variable to its corresponding integer value
    """
    val = int(np.floor((upper_bound - lower_bound + 1) * continuous) + lower_bound)
    if val < lower_bound: return int(lower_bound)
    elif val > upper_bound: return int(upper_bound)
    else: return val


def continuous_to_real(continuous: float, lower_bound: float, upper_bound: float) -> float:
    """
    Convert the continuous variable to its corresponding real value
    """
    val = (upper_bound - lower_bound) * continuous + lower_bound
    if val < lower_bound: return lower_bound
    elif val > upper_bound: return upper_bound
    else: return val


class Space:
    def __init__(self, inputs: dict):
        self.labels = list(inputs.keys())
        self.dims = len(self.labels)
        self.bounds = np.array([[0., 1.] for _ in range(self.dims)])  # opt bounds
        self.data = None
        self.types = np.array([value[0] for key, value in inputs.items()])
        self.type_bnds = np.array([value[1] for key, value in inputs.items()])

    def init_container(self, length):
        self.data = np.zeros((length, self.dims+1))

    def update_container(self, row: Union[int, List, np.ndarray], value: Union[List, np.ndarray]):
        if self.data is None: raise Exception("Data container must be initialized first with: 'init_container'")
        self.data[row] = value

    def get_nonzero(self):
        if self.data is None: raise Exception("Data container must be initialized first with: 'init_container'")
        return self.data[~np.all(self.data == 0, axis=1)]

    def save(self, filename):
        np.savetxt(filename, self.data, delimiter=',')

    def transform(self, inputs):
        def trsf(x):
            trsf_inputs = []
            for i, t in enumerate(self.types):
                if t == float:
                    trsf_inputs.append(continuous_to_real(x[i], self.type_bnds[i][0], self.type_bnds[i][1]))
                else:
                    trsf_inputs.append(continuous_to_int(x[i], self.type_bnds[i][0], self.type_bnds[i][1]))
            return trsf_inputs

        inputs = inputs.reshape(1, -1).copy() if inputs.ndim == 1 else inputs.copy()
        transformed = np.array([trsf(x) for x in inputs])
        return transformed
