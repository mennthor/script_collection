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
