"""
Implements anomaly detection based on conformance scores with application to streamed data
using path signatures.

Reference:
Cochrane, T., Foster, P., Lyons, T., & Arribas, I. P. (2020).
Anomaly detection on streamed data. arXiv preprint arXiv:2006.03487.
"""

import itertools
from joblib import Memory, Parallel, delayed

import iisignature
import numpy as np
from tqdm import tqdm

from shuffle import shuffle


def _sig(p, order):
    return np.r_[1., iisignature.sig(p, order)]


def _get_basis(dim, order):
    alphabet = range(dim)

    basis = [itertools.product(*([alphabet] * n)) for n in range(order + 1)]
    basis = list(itertools.chain(*basis))

    return basis


def _build_row(w, basis, E):
    Ai = []
    for v in basis:
        z = shuffle(w, v)
        Ai.append(sum(E[z_] for z_ in z))

    return np.array(Ai)

def _build_matrix(basis, E, enable_progress=True, **parallel_kwargs):
    A = np.zeros((len(basis), len(basis)))

    pbar = tqdm(basis, total=len(basis), disable=not enable_progress,
                desc="Building shuffle matrix")
    A = np.array(Parallel(**parallel_kwargs)(delayed(_build_row)(w, basis, E) for w in pbar))

    A_inv = np.linalg.pinv(A)

    return A_inv

def _prepare(corpus, order, enable_progress=True, **parallel_kwargs):
    dim = corpus[0].shape[1]
    basis = _get_basis(dim, order)
    basis_extended = _get_basis(dim, 2 * order)

    sigs = np.array(Parallel(**parallel_kwargs)(delayed(_sig)(p, 2 * order)
                                                for p in tqdm(corpus,
                                                              disable=not enable_progress,
                                                              desc="Computing signatures")))
    E = dict(zip(basis_extended, np.mean(sigs, axis=0)))

    A_inv = _build_matrix(basis, E, enable_progress, **parallel_kwargs)

    return sigs, A_inv


def variance(paths, corpus, order, cache_dir=None,
             enable_progress=True, **parallel_kwargs):
    r"""
    Compute conformance scores for streams in a testing collection, given a collection of
    streams in a training corpus and based on using signatures of a specified order as the
    feature map. Caches results on disk.

    Parameters
    ----------
    paths: iterable
        Collection of streams for which to compute comformance scores. Each element in the
        collection is an N_i x M array, where N_i is the number of observations in the ith
        stream and where M is dimensionality of each observation.
    corpus: iterable
        Collection of streams forming the training corpus. Each element in the collection
        is an N_i x M array, where N_i is the number of observations in the ith stream
        and where M is dimensionality of each observation.
    order: int
        Desired signature order
    cache_dir: str
        Directory for caching results of the function call
        (defaults to None, which disables caching)
    enable_progress: bool
        Whether to enable tqdm progress bars (defaults to True)
    \**parallel_kwargs:
        Additional keyword arguments (e.g. n_jobs) are passed to joblib.Parallel, thus
        influencing parallel execution. For additional information, please refer to
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        Parallel execution is disabled by default.
    """
    sigs, A_inv = Memory(cache_dir, verbose=0).cache(_prepare)(corpus, order,
                                                               enable_progress,
                                                               **parallel_kwargs)

    res = []
    for path in tqdm(paths, disable=not enable_progress, desc="Computing variances"):
        sig = _sig(path, order)
        a = sig - sigs[:, :len(sig)]
        res.append(np.diag(np.dot(a, A_inv).dot(a.T)).min())

    return res
