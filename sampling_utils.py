import numpy as np
import scipy.optimize as opt
import utils
from bisect import bisect
import logging

def gumbel_max_sample(x):
    """
    x: log-probability distribution (unnormalized is ok) over discrete random variable
    """
    
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return np.nanargmax(x + z)

def exponential_sample(x):
    """
    probability distribution over discrete random variable
    """
    E = -np.log(np.random.uniform(size=len(x)))
    E /= x
    return np.nanargmin(E)

def log_multinomial_sample(x):
    """
    x: log-probability distribution (unnormalized is ok) over discrete random variable
    """
    x[np.where(np.isnan(x))] = utils.NEG_INF
    c = np.logaddexp.accumulate(x) 
    key = np.log(np.random.uniform())+c[-1]
    return bisect(c, key)

def log_nucleus_multinomial_sample(x, size=1, nucleus_p=np.log(0.95)):
    """
    x: log-probability distribution (unnormalized is ok) over discrete random variable
    """

    assert nucleus_p <= 0
    if len(x) == 1:
        return [0]*size
    inds = np.argsort(-x)
    sortedx = x[inds]
    c = np.logaddexp.accumulate(sortedx)
    last_ind = bisect(c, nucleus_p + c[-1])
    idxs = []
    for i in range(size):
        key = np.log(np.random.uniform())+c[last_ind]
        idxs.append(inds[bisect(c, key)])
    return idxs