"""
Utility functions for computing the shuffle product.
"""

from functools import lru_cache
import itertools

def concatenate(u, words, front=False):
    """Concatenates a letter with each word in an iterable."""

    for word in words:
        if front:
            yield tuple([u] + list(word))
        else:
            yield tuple(list(word) + [u])

def shuffle(w1, w2):
    """Computes the shuffle product of two words."""

    if len(w1) == 0:
        return [w2]

    if len(w2) == 0:
        return [w1]

    gen1 = concatenate(w1[-1], shuffle(w1[:-1], w2))
    gen2 = concatenate(w2[-1], shuffle(w1, w2[:-1]))

    return itertools.chain(gen1, gen2)

@lru_cache()
def halfshuffle(w1, w2):
    """Computes the half-shuffle product of two words."""

    if len(w2) == 0:
        raise ValueError("w2 cannot be the empty word.")

    return tuple(concatenate(w2[-1], shuffle(w1, w2[:-1])))
