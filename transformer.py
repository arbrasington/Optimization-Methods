# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:11:20 2022

@author: ALEXRB
"""
import numpy as np


def transform(val, var_type, bounds):
    """
    Transform a continuous variable to its real or integer value.

    Parameters
    ----------
    val : float
        Continuous value of the gene.
    var_type : class
        Type of variable for the gene.
    bounds : np.ndarray
        Lower and upper bounds in that order.

    Returns
    -------
    transformed : Optional[int, float]
        The transformed gene.

    """
    lower, upper = bounds
    
    if var_type == int:
        transformed = int(np.floor((upper - lower + 1) * val) + lower)
        if transformed > upper:  # an input value of 1.0 can cause this
            transformed = int(upper)
    else:
        transformed = (upper - lower) * val + lower
    
    return transformed