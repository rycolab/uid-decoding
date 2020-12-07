import operator
import logging
import os
import sys
from bisect import bisect_left 
from functools import reduce  
import sacrebleu

import numpy as np
from scipy.special import logsumexp

# Reserved IDs
GO_ID = 1
"""Reserved word ID for the start-of-sentence symbol. """


EOS_ID = 2
"""Reserved word ID for the end-of-sentence symbol. """


UNK_ID = 0
"""Reserved word ID for the unknown word (UNK). """


NEG_INF = -np.inf


MACHINE_EPS = np.finfo(float).eps


LOG_MACHINE_EPS = np.log(MACHINE_EPS)


INF = np.inf


EPS_P = 0.00001


def switch_to_fairseq_indexing():
    """Calling this method overrides the global definitions of the 
    reserved  word ids ``GO_ID``, ``EOS_ID``, and ``UNK_ID``
    with the fairseq indexing scheme. 
    """
    global GO_ID
    global EOS_ID
    global UNK_ID
    GO_ID = 0
    EOS_ID = 2
    UNK_ID = 3


def switch_to_t2t_indexing():
    """Calling this method overrides the global definitions of the 
    reserved  word ids ``GO_ID``, ``EOS_ID``, and ``UNK_ID``
    with the tensor2tensor indexing scheme. This scheme is used in all
    t2t models. 
    """
    global GO_ID
    global EOS_ID
    global UNK_ID
    GO_ID = 2 # Usually not used
    EOS_ID = 1
    UNK_ID = 3 # Don't rely on this: UNK not standardized in T2T


# Log summation


def log_sum_tropical_semiring(vals):
    """Approximates summation in log space with the max.
    
    Args:
        vals  (set): List or set of numerical values
    """
    return max(vals)


def log_sum_log_semiring(vals):
    """Uses the ``logsumexp`` function in scipy to calculate the log of
    the sum of a set of log values.
    
    Args:
        vals  (set): List or set of numerical values
    """
    return logsumexp(np.asarray([val for val in vals]))


log_sum = log_sum_log_semiring
"""Defines which log summation function to use. """


def oov_to_unk(seq, vocab_size, unk_idx=None):
    if unk_idx is None:
        unk_idx = UNK_ID
    return [x if x < vocab_size else unk_idx for x in seq]

# Maximum functions

def argmax_n(arr, n):
    """Get indices of the ``n`` maximum entries in ``arr``. The 
    parameter ``arr`` can be a dictionary. The returned index set is 
    not guaranteed to be sorted.
    
    Args:
        arr (list,array,dict):  Set of numerical values
        n  (int):  Number of values to retrieve
    
    Returns:
        List of indices or keys of the ``n`` maximum entries in ``arr``
    """
    if isinstance(arr, dict):
        return sorted(arr, key=arr.get, reverse=True)[:n]
    elif len(arr) <= n:
        return range(len(arr))
    elif hasattr(arr, 'is_cuda') and arr.is_cuda:
        return np.argpartition(arr.cpu(), -n)[-n:]
    return np.argpartition(arr, -n)[-n:]


def max_(arr):
    """Get indices of the ``n`` maximum entries in ``arr``. The 
    parameter ``arr`` can be a dictionary. The returned index set is 
    not guaranteed to be sorted.
    
    Args:
        arr (list,array,dict):  Set of numerical values
        n  (int):  Number of values to retrieve
    
    Returns:
        List of indices or keys of the ``n`` maximum entries in ``arr``
    """
    if isinstance(arr, dict):
        return max(arr.values())
    if isinstance(arr, list):
        return max(arr)
    return np.max(arr)


def argmax(arr):
    """Get the index of the maximum entry in ``arr``. The parameter can
    be a dictionary.
    
    Args:
        arr (list,array,dict):  Set of numerical values
    
    Returns:
        Index or key of the maximum entry in ``arr``
    """
    if isinstance(arr, dict):
        return max(arr.items(), key=operator.itemgetter(1))[0]
    else:
        return np.argmax(arr)

def flattened(X):
    """flattens list of lists"""
    return [y for x in X for y in x]

def as_ndarray(X, pad=-1, min_length=0):
    """turns list of lists into ndarray"""
    longest = max(len(max(X, key=len)), min_length)
    return np.array([i + [pad]*(longest-len(i)) for i in X])

def log1mexp_basic(x, ignore_zero=False):
    """
    Vectorizable implementation of log(1-exp(x))
    """
    if ignore_zero:
        with np.errstate(divide='ignore'):
            return np.log1p(-np.exp(x))
    return np.log1p(-np.exp(x))

def log1pexp_basic(x, ignore_zero=False):
    """
    Vectorizable implementation of log(1+exp(x))
    """
    if ignore_zero:
        with np.errstate(divide='ignore'):
            return np.log1p(np.exp(x))
    return np.log1p(np.exp(x))

def log1pexp(x):
    """
    Numerically stable implementation of log(1+exp(x)) aka softmax(0,x).

    -log1pexp(-x) is log(sigmoid(x))

    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x <= -37:
        return np.exp(x)
    elif -37 <= x <= 18:
        return np.log1p(np.exp(x))
    elif 18 < x <= 33.3:
        return x + np.exp(-x)
    else:
        return x

def log1mexp(x):
    """
    Numerically stable implementation of log(1-exp(x))

    Note: function is finite for x < 0.

    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x >= 0:
        return np.nan
    else:
        a = abs(x)
        if 0 < a <= 0.693:
            return np.log(-np.expm1(-a))
        else:
            return np.log1p(-np.exp(-a))

def log_add(x, y):
    """
    Addition of 2 values in log space.
    Need separate checks for inf because inf-inf=nan
    """
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


def log_minus(x, y):
    """
    Subtractioon of 2 values in log space.
    Need separate checks for inf because inf-inf=nan
    """
    if x == y:
        return NEG_INF
    if y > x:
        if y-x > MACHINE_EPS:
            logging.warn("Using function log_minus for invalid values")
        return np.nan
    else:
        return x + log1mexp(y-x)

def logsigmoid(x):
    """
    log(sigmoid(x)) = -log(1+exp(-x)) = -log1pexp(-x)
    """
    return -log1pexp(-x)

def signed_log_add(x, y, sign_x, sign_y):
    a,b = x,y
    sign_a, sign_b = sign_x, sign_y
    if y > x:
        a,b = y,x
        sign_a, sign_b = sign_y,sign_x

    if sign_a != sign_b:
        val = log_minus(a,b)
    else:
        val = log_add(a,b)

    return sign_a, val

def softmax(x, temperature=1.):
    return np.exp(log_softmax(x, temperature=temperature))

def log_softmax(x, temperature=1.):
    x = x/temperature
    # numerically stable log softmax
    shift_x = x - np.max(x)
    # mask invalid values (neg inf)
    b = (~np.ma.masked_invalid(shift_x).mask).astype(int)
    return shift_x - logsumexp(shift_x, b=b)

  
def binary_search(a, x): 
    i = bisect_left(a, x) 
    if i != len(a) and a[i] == x: 
        return i 
    else: 
        return -1

def perplexity(arr):
    if len(arr) == 0:
        return INF
    score = sum([s for s in arr])
    return 2**(-score/len(arr))

def prod(iterable):
    return reduce(operator.mul, iterable, 1.0)


# Functions for common access to numpy arrays, lists, and dicts
def common_viewkeys(obj):
    """Can be used to iterate over the keys or indices of a mapping.
    Works with numpy arrays, lists, and dicts. Code taken from
    http://stackoverflow.com/questions/12325608/iterate-over-a-dict-or-list-in-python
    """
    if isinstance(obj, dict):
        return obj.keys()
    else:
        return range(len(obj))


def common_iterable(obj):
    """Can be used to iterate over the key-value pairs of a mapping.
    Works with numpy arrays, lists, and dicts. Code taken from
    http://stackoverflow.com/questions/12325608/iterate-over-a-dict-or-list-in-python
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
    else:
        for index, value in enumerate(obj):
            yield index, value


def common_get(obj, key, default):
    """Can be used to access an element via the index or key.
    Works with numpy arrays, lists, and dicts.
    
    Args:
        ``obj`` (list,array,dict):  Mapping
        ``key`` (int): Index or key of the element to retrieve
        ``default`` (object): Default return value if ``key`` not found
    
    Returns:
        ``obj[key]`` if ``key`` in ``obj``, otherwise ``default``
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return obj[key] if key < len(obj) else default


def common_contains(obj, key):
    """Checks the existence of a key or index in a mapping.
    Works with numpy arrays, lists, and dicts.
    
    Args:
        ``obj`` (list,array,dict):  Mapping
        ``key`` (int): Index or key of the element to retrieve
    
    Returns:
        ``True`` if ``key`` in ``obj``, otherwise ``False``
    """
    if isinstance(obj, dict):
        return key in obj
    else:
        return key < len(obj)


# Miscellaneous


def get_path(tmpl, sub = 1):
    """Replaces the %d placeholder in ``tmpl`` with ``sub``. If ``tmpl``
    does not contain %d, return ``tmpl`` unmodified.
    
    Args:
        tmpl (string): Path, potentially with %d placeholder
        sub (int): Substitution for %d
    
    Returns:
        string. ``tmpl`` with %d replaced with ``sub`` if present
    """
    try:
        return tmpl % sub
    except TypeError:
        pass
    return tmpl


def split_comma(s, func=None):
    """Splits a string at commas and removes blanks."""
    if not s:
        return []
    parts = s.split(",")
    if func is None:
        return [el.strip() for el in parts]
    return [func(el.strip()) for el in parts]


def ngrams(sen, n):
    sen = sen.split(' ')
    output = []
    for i in range(len(sen)-n+1):
        output.append(tuple(sen[i:i+n]))
    return output

def distinct_ngrams(hypos, n):
    total_ngrams = 0
    distinct = []
    for h in hypos:
        all_ngrams = ngrams(h, n)
        total_ngrams += len(all_ngrams)
        distinct.extend(all_ngrams)
    
    if len(distinct) == 0:
        return 0
    return float(len(set(distinct)))/len(distinct)

def ngram_diversity(hypos):
    ds = [distinct_ngrams(hypos, i) for i in range(1,5)]
    return sum(ds)/4

def hamming_distance(hypo, other_hypos, pad=-1):
    if isinstance(other_hypos, np.ndarray):
        if len(hypo) != other_hypos.shape[1]:
            hypo = np.array(hypo + [pad]*(other_hypos.shape[1] - len(hypo)))
        return (hypo != other_hypos).sum()

    elif isinstance(other_hypos, list):
        distance = 0
        for h in other_hypos:
            smaller, larger = min(len(h), len(hypo)), max(len(h), len(hypo))
            distance += larger - smaller + sum([a != b for a,b 
                in zip(h[:smaller], hypo[:smaller])])
        return distance

    else:
        logging.warn("No implementation for type: "+ str(type(other_hypos)))


def sentence_bleu(sentence, reference, detokenizer=None):
    """
    Utility function for calculating sentence BLEU. 
    Expects sentence and reference as list of tokens.
    Reference may be list of multiple references
    """
    if not isinstance(reference[0], list):
        reference = [reference]
    if detokenizer is not None:
        sentence = detokenizer(sentence.split())
        reference = [detokenizer(r.split()) for r in reference]

    return sacrebleu.sentence_bleu(sentence, [reference]).score

def entropy(distribution, base=np.e):
    return -sum(distribution*np.log(distribution, base=base))

def log_entropy(log_distribution, base=np.e):
    return -sum(base**log_distribution * log_distribution)

    