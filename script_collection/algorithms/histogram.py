# coding: utf-8

"""
Methods for binning and histogramming things.
"""

import numpy as np


def get_binmids(bins):
    """
    Returns mids of a 1D bins array or mids for each bin array in a list.

    Doesn't catch any false formatted data so be sure what you do.

    Parameters
    ----------
    bins : array-like or list of array
        Either 1D array defining the bin edges for a single binning or a list
        of such arrays for multiple binnings.

    Returns
    -------
    mids : array-like or list of array
        1D array if bins was single array or list with the bin mids for each
        binning in bins. Mid arrays have on point less than the input bins.

    Example
    -------
    >>> bins = np.arange(10)
    >>> mids = get_binmids(bins)
    >>> print(mids)  # Single array
    >>> bins = [np.arange(10), np.arange(5)]
    >>> mids = get_binmids(bins)
    >>> print(mids)  # List of two arrays
    """
    def check_bins(bins):
        bins = np.asarray(bins)
        if len(bins) < 2:
            raise ValueError("A bin array must at least have two entries.")
        return bins

    if type(bins) == list:
        mids = []
        for b in bins:
            b = check_bins(b)
            mids.append(0.5 * (b[:-1] + b[1:]))
    else:
        b = check_bins(bins)
        mids = 0.5 * (b[:-1] + b[1:])

    return mids


def get_bins_from_mids(mids):
    """
    Given an 1D array of assumed center points, create bin edges around them.

    This is the inverse of ``get_binmids``.

    Parameters
    ----------
    mids : array-like or list of array
        Either 1D array defining bin mids for a single binning or a list of such
        arrays for multiple binnings.

    Returns
    -------
    bins : array-like or list of array
        1D array if mids was single array or list with the bin edges for each
        mid points in mids. Bin arrays have on point more than the input mids.

    Example
    -------
    >>> mids = np.arange(10)
    >>> bins = get_bins_from_mids(mids)
    >>> redo_mids = get_binmids(bins)
    >>> print("Mids of new edges are old mids: ", np.all(arr == redo_mids))
    """
    def make_bins(mids):
        mids = np.asarray(mids)
        if len(mids) < 2:
            raise ValueError("Doesn't work, if only a single bin mid is given")
        # First make bin centers, then mirror outermost edges and concate all
        b = 0.5 * (mids[:-1] + mids[1:])
        dl = 2 * mids[0] - b[0]
        dr = 2 * mids[-1] - b[-1]
        return np.concatenate(([dl], b, [dr]))

    if type(mids) == list:
        bins = []
        for m in mids:
            bins.append(make_bins(m))
    else:
        bins = make_bins(mids)

    return bins


def marginalize_hist(h, bins, axes):
    """
    Marginalize a given normed histogram over the given axes.

    When normed histograms are marginalized we need to sum entries with the
    respective bin width in that dimension to mimic integration over a PDF.

    Parameters
    ----------
    h : array
        ND histogram, with shape as given by np.histogrammdd.
    bins : list
        List of bins belonging to each dimension, as given by np.histogramdd.
    axes : tupel
        Which axes to reduce. Must not exceed the dimension of the hist.

    Returns
    -------
    h : array
        Marginalized histogram with now reduced dimensions.
    bins : list
        List of remaining bins in the same order as the histogram dimensions.
        If h is 1D after reduction, a single bin array is returned.
        If h is reduced to a single number, None is returned
    """
    # Get binwidths
    bins = np.array(bins)
    bw = []
    for bi in bins:
        bw.append(np.diff(bi))

    h = np.copy(h).astype(np.float64)

    # Sort axes in descending order. Otherwise the axis we want to reduce
    # might not exist after one or more reduction steps as the number of
    # dimension get reduced
    axes = np.atleast_1d(axes)
    axes = np.sort(axes)[::-1]

    # Get remaining axes after reduction to return correct bins
    rem_axes = np.arange(len(bins))
    rem = np.logical_not(np.in1d(rem_axes, axes))

    # Now reduce axes one by one. The order of the original axes is preserved
    for axi in axes:
        # First move reduction axis to last pos to match bin shape if dim >= 2
        if len(h.shape) > 1:
            h = np.moveaxis(h, axi, -1)
        # Then reduce with correct bin width
        h = np.dot(h, bw[axi])

    try:
        if len(h):
            bins = bins[rem]
        if len(bins) == 1:
            bins = bins[0]
    except TypeError:
        bins = None

    return h, bins


def histogramdd_func(sample, rvals=None, bins=10, method="count"):
    """
    Compute the multidimensional histogram of some data.
    Note: This is the same as scipy.stats.binned_stats_dd without some features
        which I found after implementing this...
        For my tests sets, this function is twice as fast however.

    Parameters
    ----------
    sample : (N, D) array, or (D, N) array_like
        The data to be histogrammed.
        Note the unusual interpretation of sample when an array_like:
        * When an array, each row is a coordinate in a D-dimensional space -
          such as ``histogramdd(np.array([p1, p2, p3]))``.
        * When an array_like, each element is the list of values for single
          coordinate - such as ``histogramdd((X, Y, Z))``.
        The first form should be preferred.
    rvals : (N, ) ndarray or None, optional (default: None)
        Data to be used to determine the value of each histogram cell. If
        `None`, `method` must be 'count'.
    bins : sequence or int, optional
        The bin specification:
        * A sequence of arrays describing the monotonically increasing bin
          edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).
    method : str or callable, optional (default: 'count')
        Can be either 'count', 'min', 'max', 'mean', 'median' or a callable
        that takes a 1D ndarray subset of `rvals` as an argument and reduces it
        to a single number, which is then stored in the histogram.
        Eg. giving 'max' is equivalent to `lambda x: np.max(x)`.
        If 'count' is given, values in `rvals` are ignored, because they are
        simply counted and the method acts as the usual `np.histogramdd`.

    Returns
    -------
    H : ndarray
        The multidimensional histogram of sample x. It has shape
        `(len(bins) - 1 for bins in edges)` per dimension.
    edges : list
        A list of D arrays describing the bin edges for each dimension.
    indices : ndarray
        Array of same shape as `H` having a list of indices of input points
        falling into each bin referenced by `H`.

    Examples
    --------
    >>> r = np.random.randn(100, 3)
    >>> H, edges = histogramdd_func(r, bins = (5, 8, 4), method='count')
    >>> H.shape, edges[0].size, edges[1].size, edges[2].size
    ((5, 8, 4), 6, 9, 5)
    >>> np.all(H == np.histogramdd(r, bins = (5, 8, 4))))
    True
    >>> r.max() == histogramdd_func(r, bins = (5, 8, 4), method='max')[0].max()
    True
    >>> r.min() == histogramdd_func(r, bins = (5, 8, 4), method='min')[0].min()
    True
    """
    # We basically steal everything from numpy.histogrammdd
    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    if rvals is None:
        if method != "count":
            raise ValueError("`method` is not 'count' but `rvals` is None.")
    else:
        rvals = np.atleast_1d(rvals)
        if len(rvals.shape) != 1 or len(rvals) != N:
            raise ValueError("'rvals' array must be a 1D array of length N.")

    nbin = np.empty(D, int)
    edges = D * [None]
    dedges = D * [None]

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                "The dimension of bins must be equal to the dimension of the "
                " sample x.")
    except TypeError:
        # bins is an integer
        bins = D * [bins]

    # Normalize the range_ argument
    range_ = (None,) * D

    # Create edge arrays
    for i in range(D):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    "`bins[{}]` must be positive, when an integer".format(i))
            smin, smax = _get_outer_edges(sample[:,i], range_[i])
            try:
                n = operator.index(bins[i])

            except TypeError as e:
                raise TypeError(
                    "`bins[{}]` must be an integer, when a scalar".format(i)
                ) from e

            edges[i] = np.linspace(smin, smax, n + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError("`bins[{}]` must be monotonically "
                                 "increasing, when an array".format(i))
        else:
            raise ValueError(
                "`bins[{}]` must be a scalar or 1d array".format(i))

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
        dedges[i] = np.diff(edges[i])

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side="right")
        for i in range(D)
    )

    # Using searchsorted, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute the given metric for all samples falling into each bin and assign
    # it to the flattened histmat.
    if method == "count":
        hist = np.bincount(xy, minlength=nbin.prod())
    else:
        # Select the method to reduce all cell entries
        if isinstance(method, str):
            if method == "min":
                _reduce = np.min
            elif method == "max":
                _reduce = np.max
            elif method == "mean":
                _reduce = np.mean
            elif method == "median":
                _reduce = np.median
            else:
                raise ValueError("If str, 'method' can be one of 'count', "
                                 "'min', max', 'mean', 'median'.")
        elif isinstance(method, callable):
            _reduce = method
        else:
            raise TypeError("'methods' must be either a str or a callable, "
                            "'{}' was given.".format(type(method)))

        # We want to obtain a reasonably quick way to group all samples per bin.
        # np.bincount does it in C, by allocating a new array and simply adding
        # the indices in a single pass. Here we can speed things up by using
        # binary search to obtain all places in xy where a new index is
        # introduced, avoiding masking the entire array over and over.
        hist = np.zeros(nbin.prod(), dtype=float) * np.nan
        xy_srt_idx = np.argsort(xy)
        xy_srt = xy[xy_srt_idx]
        xy_uni = np.unique(xy_srt)
        idx_hi = np.searchsorted(xy_srt, xy_uni, side="right")
        idx_hi = np.r_[0, idx_hi]
        for j, (lo, hi) in enumerate(zip(idx_hi[:-1], idx_hi[1:])):
            hist[xy_uni[j]] = _reduce(rvals[xy_srt_idx[lo:hi]])

    # Shape into a proper matrix
    hist = hist.reshape(nbin)

    # This preserves the (bad) behavior observed in gh-7845, for now.
    hist = hist.astype(float, casting="safe")

    # Remove outliers (indices 0 and -1 for each dimension).
    core = D * (slice(1, -1),)
    hist = hist[core]

    if (hist.shape != nbin - 2).any():
        raise RuntimeError("Internal Shape Error")
    return hist, edges
