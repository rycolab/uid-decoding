# Modified version of Tim Vieira's SumHeap https://github.com/timvieira/arsenal
# cython: language_level=3, boundscheck=False, infer_types=True, nonecheck=False
# cython: overflowcheck=False, initializedcheck=False, wraparound=False, cdivision=True
import numpy as np

from libc.stdlib cimport rand
from libc.math cimport log2, ceil
cdef extern from "limits.h":
    int INT_MAX

ctypedef double (*f_type)(double, double)
cdef double MAX = float(INT_MAX)
cdef double NEG_INF = -np.inf
cdef inline double uniform() nogil:
    return rand() / MAX

cdef class SumHeap:
    cdef f_type add, minus, div, mult
    cdef readonly:
        double[:] S
        int n, d
        double zero
        bint log_space

    def __init__(self, double[:] w, bint log_space=False):
        self.log_space = log_space
        if self.log_space:
            self.zero = NEG_INF 
            self.add, self.minus, self.div, self.mult = logadd, logminus, logdiv, logmult
        else:
            self.zero = 0.
            self.add, self.minus, self.div, self.mult = add, minus, div, mult
        
        self.n = w.shape[0]
        self.d = int(2**ceil(log2(self.n)))   # number of intermediates
        self.S = np.full(2*self.d, self.zero)           # intermediates + leaves
        self.heapify(w)

    def __getitem__(self, int k):
        return self.S[self.d + k]

    def __setitem__(self, int k, double v):
        self.update(k, v)

    cpdef void heapify(self, double[:] w):
        "Create sumheap from weights `w` in O(n) time."
        d = self.d; n = self.n
        self.S[d:d+n] = w                         # store `w` at leaves.
        for i in reversed(range(1, d)):
            self.S[i] = self.add(self.S[2*i], self.S[2*i + 1])

    cpdef void update(self, int k, double v):
        "Update w[k] = v` in time O(log n)."
        i = self.d + k
        self.S[i] = v
        while i > 0:   # fix parents in the tree.
            i //= 2
            self.S[i] = self.add(self.S[2*i], self.S[2*i + 1])

    cpdef int sample(self, u=None):
        "Sample from sumheap, O(log n) per sample."
        cdef double left, p
        if u is None: u = np.log(uniform()) if self.log_space else uniform()
        d = self.S.shape[0]//2     # number of internal nodes.
        p = self.mult(u, self.S[1])  # random probe, p ~ Uniform(0, z)
        # Use binary search to find the index of the largest CDF (represented as a
        # heap) value that is less than a random probe.
        i = 1
        while i < d:
            # Determine if the value is in the left or right subtree.
            i *= 2            # Point at left child
            left = self.S[i]  # Probability mass under left subtree.
            if p > left:      # Value is in right subtree.
                p = self.minus(p, left)     # Subtract mass from left subtree
                i += 1        # Point at right child
        return i - d

    cpdef long[:] swor(self, int k):
        "Sample without replacement `k` times."
        cdef long[:] z = np.zeros(k, dtype=int)
        for i in range(k):
            k = self.sample()
            z[i] = k
            self.update(k, self.zero)
        return z


cdef inline double add(double x, double y):
    return x+y
cdef inline double minus(double x, double y):
    return x-y
cdef inline double div(double x, double y):
    return x/y
cdef inline double mult(double x, double y):
    return x*y
cdef inline double logdiv(double x, double y):
    return x-y
cdef inline double logmult(double x, double y):
    return x+y

cdef double logadd(double x, double y):
    if x == NEG_INF:
        return y
    elif y == NEG_INF:
        return x
    else:
        if y <= x:
            d = y-x
            r = x
        else:
            d = x-y
            r = y
        return r + log1pexp(d)

cdef double logminus(double x, double y):
    if x == y:
        return NEG_INF
    if y > x:
        return np.nan
    else:
        return x + log1mexp(y-x)


cdef double log1pexp(double x):
    if x <= -37:
        return np.exp(x)
    elif -37 <= x <= 18:
        return np.log1p(np.exp(x))
    elif 18 < x <= 33.3:
        return x + np.exp(-x)
    else:
        return x

cdef double log1mexp(double x):
    if x >= 0:
        return np.nan
    else:
        a = abs(x)
        if 0 < a <= 0.693:
            return np.log(-np.expm1(-a))
        else:
            return np.log1p(-np.exp(-a))

