# coding: utf-8

"""
ndarray utility methods.
"""

import numpy as np

from ..algorithms.math import get_perms


def idx2rowcol(idx, ncols):
    """
    Convert a 1d running index to ``[row, col]`` indices. ``numpy`` broadcasting
    rules apply to ``idx``

    Parameters
    ----------
    idx : int or array-like
        Current index / indices, ``idx >= 0``.
    ncols : int
        Number of columns to index, ``ncols > 0``.

    Returns
    -------
    row, col : int
        Row and column indices.
    """
    row = np.floor(idx / ncols).astype(int)
    col = np.mod(idx, ncols).astype(int)
    return row, col


def minmax(a, axis=None):
    """
    Return minimum and maximum values from a given array.

    Parameters
    ----------
    a : array_like
        Input array, can have multiple axes.
    axis : None or int, optional
        Axis along which to operate. If None flattened input is used.
        (default: None)

    Returns
    -------
    minmax : tuple, shape (2)
        (min, max) values along the selected axis of the input array.
    """
    return np.amin(a, axis=axis), np.amax(a, axis=axis)


def is_sorted(arr, order="ascending"):
    """
    Check if an 1D array is sorted.

    https://stackoverflow.com/questions/3755136

    Parameters
    ----------
    arr : array-like
        1D array which supports sorting operations on it's elements.
    order : string, optional
        Sorting order to check, can be ['ascending'|'descending'].
        (default: 'ascending')

    Retruns
    -------
    sorted : bool
        True, if the array is sorted on the given oder, False if not.
    """
    if order == "descending":
        arr = arr[::-1]
    # Test if arr is sorted in ascending order
    for i, item in enumerate(arr[1:]):
        if item < arr[i]:
            return False
    return True


def chunker(arr, size):
    """
    Split array into chunks with given size, last bit may be shorter.

    From Stackoverflow: 434287, nosklo :+1:

    Parameters
    ----------
    arr : array-like, shape (n_samples, n_features)
        Array to be split in chunks.
    size : int
        Number of elements in each chunk.

    Returns
    -------
    chunks : iterator, length int(ceil(len(arr) / size))
        Iterator returning a new chunk with length `size` in each
        iteration. The last element has a smaller length, if
        len(arr) % size != 0.
    """
    return (arr[pos:pos + size] for pos in range(0, len(arr), size))


def reflect_at_bounds(arr, bounds):
    """
    Reflect array points at given borders.

    Parameters
    ----------
    arr : array-like, shape (n_samples, n_features)
        Data points. Each point is a row, each feature a column.
    borders : array-like, shape (n_features, 2)
        [low, hig] border for each dimension.

    Returns
    -------
    ref : array-like, shape (3**n_features - 1, n_samples, n_features)
        Original and reflected points
    """
    # Get all directions to reflect into
    n_features = arr.shape[1]
    perms = get_perms(n_features)
    n_perms = len(perms)

    # Create reflected array prototype in shape (n_perms, n_samples, n_features)
    ref = np.repeat([arr], repeats=n_perms, axis=0)

    # Now just loop over perms and fill points. Three cases in each dimension:
    # - If direction is  0: leave original point from old array
    # - If direction is +1: Add high border to orignal values and fill these
    # - If direction is -1: Subtract point from low border and fill these
    for i, perm in enumerate(perms):
        for ax, edge in enumerate(perm):
            if edge == 0:  # In orginal range, don't mirror
                pass

            if edge == +1:  # Mirror at higher boundary to higher values
                ref[i, :, ax] = 2. * bounds[ax, 1] - ref[i, :, ax]

            if edge == -1:  # Mirror at lower boundary to lower values
                ref[i, :, ax] = 2. * bounds[ax, 0] - ref[i, :, ax]

    return ref


def diag_indices_ndim(nelems, ndim, k=1):
    """
    Returns indices to select diagonal and off-diagonal items in a ND square
    matrix with `nelems` elements per dimension (side-length).

    Parameters
    ----------
    nelems : int
        Side length of the matrix. It's assumed to be square so the side-length
        applies for alll dimensions.
    ndim : int
        Number of dimensions of the matrix.
    k : int, optional (default: 1)
        Width of off-diagonal. Selects all elements that are no further than
        `k` away from any diagonal index in absolute norm
        `sum(abs(idx - idx_diag))`.

    Returns
    -------
    idx : ndarray
        Shape (nvals, ndim). Each row holds the indes for each dimension to
        select one off-diagonal element.
        All diagonal indices `(j, ..., j)` are also included.

    Example
    -------
    diag_indices_ndim(3, ndim=2, k=1)
    >>> array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]])
    idx = diag_indices_ndim(10, ndim=2, k=1)
    plt.scatter(idx[:, 0], idx[:, 1])
    plt.show()
    """
    # First create all valid combination we can travel from each diag point
    off_idx = np.array(list(product(*[np.arange(-k, k + 1) for _ in range(ndim)])))
    # Using the taxi norm, we can select the proper no. of off diags
    taxi = np.sum(np.abs(off_idx), axis=1)
    off_idx = off_idx[taxi <= k]

    # Get the strating points aka diagonal indices per dim
    diag_idx = np.diag_indices(nelems, ndim=ndim)

    # Combine into unique, in-bound absolute indices
    abs_idx = np.concatenate([didx + off_idx for didx in list(zip(*diag_idx))])
    abs_idx = np.unique(abs_idx, axis=0)
    return abs_idx[~np.any((abs_idx < 0) | (abs_idx > nelems - 1), axis=1)]
