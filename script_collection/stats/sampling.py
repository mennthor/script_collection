# coding: utf-8

"""
Contains random sampling methods.
"""
import numpy as np
import scipy.optimize as sco
from sklearn.utils import check_random_state


def random_choice(rndgen, CDF, size=None):
    """
    Stripped implementation of ``np.random.choice`` without the checks for the
    weights making it significantly faster. If ``CDF`` is not a real CDF this
    will produce non-sense.

    Note: In general the CDF is built by cumulative summing up event weights.
          For unweighted events, the weights are all equal.
    Note: Only samples with replacement.

    Parameters
    ----------
    rndgen : np.random.RandomState instance
        Random state instance to sample from.
    CDF : array-like
        Correctly normed CDF used to sample from. ``CDF[-1]=1`` and
        ``CDF[i-1]<=CDF[i]``.
    size : int or None, optional
        How many indices to sample. If ``None`` a single int is returned.
        (default: ``None``)

    Returns
    –------
    idx : int or array-like
        The sampled indices of the chosen CDF values. Can be inserted in the
        original array to obtain the values.
    """
    u = rndgen.uniform(size=size)
    return np.searchsorted(CDF, u, side="right")


def random_choice_2d(rndgen, CDFs, size_per_cdf=1):
    """
    Stripped 2D implementation of ``np.random.choice`` without the checks for
    the weights. If ``CDF`` is not a real CDF this will produce non-sense.

    Note: In general the CDF is built by cumulative summing up event weights.
          For unweighted events, the weights are all equal.
    Note: Only samples with replacement.

    Parameters
    ----------
    rndgen : np.random.RandomState instance
        Random state instance to sample from.
    CDFs : array-like, shape (ncdfs, len(cdf_i))
        Correctly normed CDFs used to sample from. ``CDF[:, -1] = 1`` and
        ``CDF[:, i-1] <= CDF[:, i]``.
    size_per_cdf : int , optional
        How many indices to sample per CDFs. (default: 1)

    Returns
    –------
    idx : array-like, shape (ncdfs, size_per_cdf)
        The sampled indices for each CDF. Can be inserted per row in the
        original array to obtain the sampled values.
    """
    if size_per_cdf is None:
        size_per_cdf = 1

    ncdfs, len_cdfs = CDFs.shape
    # Get random nums for each CDF with given size per PDF
    u = rndgen.uniform(size=(ncdfs, size_per_cdf))
    # Add unique offset for each CDF
    offset = np.arange(0, 2 * ncdfs, 2, dtype=int)[:, None]
    u = np.ravel(u + offset)
    # Flatten offset CDFs to searchsorted all at once
    cdfs_flat = np.ravel(CDFs + offset)
    # Reshape indices to match the original CDFs again
    idx = np.searchsorted(cdfs_flat, u, side="right")
    return idx.reshape(ncdfs, size_per_cdf) - offset // 2 * len_cdfs


def random_choice_2d_looped(rndgen, CDFs, size_per_cdf=None):
    """
    Same as ``random_choice_2d`` but with a loop instead of broadcasting. The
    broadcasted version is faster for many CDFs, but takes more memory.
    """
    idx = np.empty((len(CDFs), size_per_cdf), dtype=int)
    for i, cdf_i in enumerate(CDFs):
        u = rndgen.uniform(size=size_per_cdf)
        idx[i] = np.searchsorted(cdf_i, u, side="right")
    return idx


def rejection_sampling(pdf, bounds, n, fmax=None, random_state=None):
    """
    Generic rejection sampling method for ND pdfs with f: RN -> R.
    The ND function `pdf` is sampled in intervals xlow_i <= func(x_i) <= xhig_i
    where i=1, ..., N and n is the desired number of events.

    1. Find maximum of function. This is our upper boundary fmax of f(x)
       Only if ``fmax`` is ``None``, otherwise use ``fmax``as upper border.
    2. Loop until we have n valid events, start with m = n
        1. Create m uniformly distributed Ndim points in the Ndim bounds.
           Coordinates of these points are
           r1 = [[x11, ..., x1N ], ..., [xm1, ..., xmN]]
        2. Create m uniformly distributed numbers r2 between 0 and fmax
        3. Calculate the pdf value pdf(r1) and compare to r2
           If r2 <= pdf(r1) accept the event, else reject it
        4. Append only accepted random numbers to the final list

    This generates points that occur more often (or gets less rejection) near
    high values of the pdf and occur less often (get more rejection) where the
    pdf is small.

    To maximize efficiency the upper boundary for the random numbers is the
    maximum of the function over the defined region.

    Parameters
    ----------
    func : function
        Function from which to sample. func is taking exactly one argument 'x'
        which is a n-dimensional array containing a N dimensional point in each
        entry (as in scipy.stats.multivariat_normal):
            x = [ [x11, ..., x1N], [x21, ..., x2N], ..., [xn1, ..., xnN]
        func must be >= 0 everywhere.
    xlow, xhig : array-like, shape (ndims, 2)
        Arrays with the rectangular borders of the pdf. The length of xlow and
        xhig must be equal and determine the dimension of the pdf.
    n : int
        Number of events to be sampled from the given pdf.
    fmax : float, optional
        If not ``None``, use the given value as the upper function value for the
        sampler. Choosing this too low for the PDF results in incorrect
        sampling. If ``None``a fitter tries to find the maximum function value
        before the sampling step. (default: ``None``)
    random_state : seed, optional
        Turn seed into a np.random.RandomState instance. See
        `sklearn.utils.check_random_state`. (default: None)

    Returns
    -------
    sample : array-like
        A list of the n sampled points
    eff : float
        The efficiency (0 < eff < 1) of the sampling, how many random numbers
        were rejected before the desired n samples were drawn.
    """
    if n == 0:
        return np.array([], dtype=np.float), 1.
    bounds = np.atleast_2d(bounds)
    dim = bounds.shape[0]

    rndgen = check_random_state(random_state)

    def negpdf(x):
        """To find the maximum we need to invert to use scipy.minimize."""
        return -1. * pdf(x)

    def _pdf(x):
        """PDF must be positive everywhere, so raise error if not."""
        res = pdf(x)
        if np.any(res < 0.):
            raise ValueError("Evaluation of PDF resultet in negative value.")
        return res

    xlow, xhig = bounds[:, 0], bounds[:, 1]
    if fmax is None:
        # Random scan the PDF to get a (hopefully) good seed for the fit
        ntrials = 10**dim
        x_scan = rndgen.uniform(xlow, xhig, size=(ntrials, dim))
        min_idx = np.argmin(negpdf(x_scan))
        x0 = x_scan[min_idx]
        xmin = sco.minimize(negpdf, x0, bounds=bounds, method="L-BFGS-B").x
        fmax = _pdf(xmin)

    # Draw remaining events until all n samples are created
    sample = []
    nstart = n
    efficiency = 0
    while n > 0:
        # Count trials for efficiency
        efficiency += n

        r1 = (xhig - xlow) * rndgen.uniform(
            0, 1, size=n * dim).reshape(dim, n) + xlow
        r2 = fmax * rndgen.uniform(0, 1, size=n)

        accepted = (r2 <= _pdf(r1))
        sample += r1[accepted].tolist()

        n = np.sum(~accepted)

    # Efficiency is (generated events) / (all generated events)
    efficiency = nstart / float(efficiency)
    return np.array(sample), efficiency


def power_law_sampler(gamma, xlow, xhig, n, random_state=None):
    r"""
    Sample n events from a power law with index gamma between xlow and xhig
    by using the analytic inversion method. The power law pdf is given by

    .. math::
       \mathrm{pdf}(\gamma) = x^{-\gamma} / \mathrm{norm}

    where norm ensures an area under curve of one. Positive spectral index
    gamma means a falling spectrum.

    Note: When :math:`\gamma=1` the integral is

    .. math::
       \int 1/x \mathrm{d}x = ln(x) + c

    This case is also handled.

    Sampling of power laws over multiple order of magnitude with the rejection
    method is VERY inefficient.

    Parameters
    ----------
    gamma : float
        Power law index.
    xlow, xhig : float
        Border of the pdf, needed for proper normalization.
    n : int
        Number of events to be sampled.
    random_state : seed, optional
        Turn seed into a np.random.RandomState instance. See
        `sklearn.utils.check_random_state`. (default: None)

    Returns
    -------
    sample : float array
        Array with the requested n numbers drawn distributed as a power law
        with the given parameters.
    """
    rndgen = check_random_state(random_state)
    # Get uniform random number which is put in the inverse function
    u = rndgen.uniform(size=int(n))

    if gamma == 1:
        return np.exp(u * np.log(xhig / xlow)) * xlow
    else:
        radicant = (u * (xhig**(1. - gamma) - xlow**(1. - gamma))
                    + xlow**(1. - gamma))
        return radicant**(1. / (1. - gamma))
