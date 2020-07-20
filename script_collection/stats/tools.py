# coding: utf8

"""
Different tools for different purposes.
"""
import numpy as np
import scipy.stats as scs

from .sampling import random_choice


def sigma2prob(sig):
    """
    Return the probability for a given gaussian sigma central interval.
    Reverse operation of ``prob2sigma``.
    """
    sig = np.atleast_1d(sig)
    return scs.norm.cdf(sig) - scs.norm.cdf(-sig)  # Central interval


def prob2sigma(p):
    """
    Return the corresponding sigma for an assumed total probability of
    ``1-p`` in both tails of a gaussian PDF - so ``p/2`` in each of the tails.
    Reverse operation of ``sigma2prob``

    Example
    -------
    >>> sig = np.array([1, 2, 3, 4, 5])
    >>> p = sigma2prob(sig)
    >>> np.allclose(prob2sigma(p), sig)
    """
    p = np.atleast_1d(p)
    return scs.norm.ppf(p + (1. - p) / 2.)  # (1-p)/2 in the right tail


def standardize_nd_sample(sam, mean=None, cov=None,
                          cholesky=True, ret_stats=False, diag=False):
    r"""
    Standardizes a n-dimensional sample using the Mahalanobis distance.

    .. math:: x' = \Sigma^{-1/2} (x - y)

    The resulting sample :math:`x'` has a zero mean vector and an identity
    covariance.

    Parameters
    ----------
    sam : array-like, shape (n_samples, n_features)
        Data points in the sample, each column is a feature, each row a point.
    mean : array-like, shape (n_features), optional
        If explicitely given, use this mean vector for the transformation. If
        None, the estimated mean from data is used. (default: None)
    cov : array-like, shape (n_features, n_features), optional
        If explicitely given, use this covariance matrix for the transformation.
        If None, the estimated cov from data is used. (default: None)
    cholesky : bool, optional
        If true, use fast Cholesky decomposition to calculate the sqrt of the
        inverse covariance matrix. Else use eigenvalue decomposition (Can be
        numerically unstable, not recommended). (default: True)
    ret_stats : bool, optional
        If True, the mean vector and covariance matrix of the input sample are
        returned, too. (default: False)
    diag : bool
        If True, only scale by variance, diagonal cov matrix. (default: False)

    Returns
    -------
    stand_sam : array-like, shape (n_samples, n_features)
        Standardized sample, with mean = [0., ..., 0.] and cov = identity.

    Optional Returns
    ----------------
    mean : array-like, shape(n_features)
        Mean vector of the input data, only if ret_stats is True.
    cov : array-like, shape(n_features, n_features)
        Covariance matrix of the input data, only if ret_stats is True.

    Example
    -------
    >>> mean = [10, -0.01, 1]
    >>> cov = [[14, -.2, 0], [-.2, .1, -0.1], [0, -0.1, 1]]
    >>> sam = np.random.multivariate_normal(mean, cov, size=1000)
    >>> std_sam = standardize_nd_sample(sam)
    >>> print(np.mean(std_sam, axis=0))
    >>> print(np.cov(std_sam, rowvar=False))
    """
    if len(sam.shape) != 2:
        raise ValueError("Shape of `sam` must be (n_samples, n_features).")
    if mean is None and cov is None:
        # Mean and cov over the first axis
        mean = np.mean(sam, axis=0)
        cov = np.atleast_2d(np.cov(sam, rowvar=False))
    elif mean is not None and cov is not None:
        mean = np.atleast_1d(mean)
        cov = np.atleast_2d(cov)
        if len(mean) != sam.shape[1]:
            raise ValueError("Dimensions of mean and sample don't match.")
        if cov.shape[0] != sam.shape[1]:
            raise ValueError("Dimensions of cov and sample don't match.")

    if diag:
        cov = np.diag(cov) * np.eye(cov.shape[0])

    if cholesky:
        # Cholesky produces a tridiagonal matrix from A with: L L^T = A
        # To get the correct trafo, we need to transpose the returned L:
        #   L.L^t
        sqrtinvcov = np.linalg.cholesky(np.linalg.inv(cov)).T
    else:
        # The naive sqrt of eigenvalues. Is (at least) instable for > 3d
        # A = Q lam Q^-1. If A is symmetric: A = Q lam Q^T
        lam, Q = np.linalg.eig(np.linalg.inv(cov))
        sqrtlam = np.sqrt(lam)
        sqrtinvcov = np.dot(sqrtlam * Q, Q.T)

    # Transform each sample point and reshape result (n_samples, n_features)
    stand_sam = np.dot(sqrtinvcov, (sam - mean).T).T

    if ret_stats:
        return stand_sam, mean, cov
    else:
        return stand_sam


def shift_and_scale_nd_sample(sam, mean, cov, cholesky=True):
    r"""
    Shift and scale a nD sample by given mean and covariance matrix.

    This is the inverse operation of `standardize_nd_sample`. If a
    standardized sample :math:`x'` with zero mean vector and identity covariance
    matrix is given, it is rescaled and shifted using

    .. math:: x = (\Sigma^{1/2} x) + y

    then having a mean vector `mean` and a covariance matrix `cov`.

    Parameters
    ----------
    sam : array-like, shape (n_samples, n_features)
        Data points in the sample, each column is a feature, each row a point.
    mean : array-like, shape (n_features)
        Mean vector used for the transformation.
    cov : array-like, shape (n_features, n_features)
        Covariance matrix used for the transformation.

    Returns
    -------
    scaled_sam : array-like, shape (n_samples, n_features)
        Scaled sample using the transformation with the given mean and cov.
    """
    if len(sam.shape) != 2:
        raise ValueError("Shape of `sam` must be (n_samples, n_features).")
    mean = np.atleast_1d(mean)
    cov = np.atleast_2d(cov)
    if len(mean) != sam.shape[1]:
        raise ValueError("Dimensions of mean and sample don't match.")
    if cov.shape[0] != sam.shape[1]:
        raise ValueError("Dimensions of cov and sample don't match.")

    # Transformation matrix: inverse of original trafo
    sqrtinvcov = np.linalg.cholesky(np.linalg.inv(cov)).T
    sqrtcov = np.linalg.inv(sqrtinvcov)

    return np.dot(sqrtcov, sam.T).T + mean


def weighted_cdf(x, val, weights=None):
    """
    Calculate the weighted CDF of data ``x`` with weights ``weights``.

    This calculates the fraction  of data points ``x <= val``, so we get a CDF
    curve when ``val`` is scanned for the same data points.
    The uncertainty is calculated from weighted binomial statistics using a
    Wald approximation.

    Inverse function of ``weighted_percentile``.

    Parameters
    ----------
    x : array-like
        Data values on which the percentile is calculated.
    val : float
        Threshold in x-space to calculate the percentile against.
    weights : array-like
        Weight for each data point. If ``None``, all weights are assumed to be
        1. (default: None)

    Returns
    -------
    cdf : float
        CDF in ``[0, 1]``, fraction of ``x <= val``.
    err : float
        Estimated uncertainty on the CDF value.

    Note
    ----
    https://stats.stackexchange.com/questions/159204/how-to-calculate-the-standard-error-of-a-proportion-using-weighted-data
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval
    """
    x = np.asarray(x)

    if weights is None:
        weights = np.ones_like(x)
    elif np.any(weights < 0.) or (np.sum(weights) <= 0):
        raise ValueError("Weights must be positive and add up to > 0.")

    # Get weighted CDF: Fraction of weighted values in x <= val
    mask = (x <= val)
    weights = weights / np.sum(weights)
    cdf = np.sum(weights[mask])
    # Binomial error on weighted cdf in Wald approximation
    err = np.sqrt(cdf * (1. - cdf) * np.sum(weights**2))

    return cdf, err


def _weighted_cdf(x, vals, weights=None, sorted=False, err="wald",
                  cl=0.68, n_bootstraps=50):
    """
    TODO: Replace the above weighted_cdf method with this one.
          Include CL errors instead of std to bootstrap error estimation.
          Include Clopper Pearson or Wilson intervals.
    """
    x = np.atleast_1d(x)
    vals = np.atleast_1d(vals)

    if weights is None:
        weights = np.ones_like(x)
    elif np.any(weights < 0.) or (np.sum(weights) <= 0):
        raise ValueError("Weights must be positive and add up to > 0.")
    weights = np.atleast_1d(weights) / np.sum(weights)

    if not sorted:
        srt_idx = np.argsort(x)
        x, weights = x[srt_idx], weights[srt_idx]

    idx = np.searchsorted(x, vals, side="right")
    cdfs = np.array([np.sum(weights[:i]) for i in idx])

    alpha = 1. - cl
    if err == "wald":
        # Binomial error on weighted cdf in Wald approximation
        errs = (scs.norm.cdf(1. - alpha / 2.)
                * np.sqrt(cdfs * np.clip((1. - cdfs), 0, 1)
                          * np.sum(weights**2)))
    elif err == "bootstrap":
        # Do a bootstrap error estimate
        CDF = np.cumsum(weights)
        assert np.isclose(CDF[-1], 1.)
        size = len(x)
        _cdfs = []
        for i in range(n_bootstraps):
            _x = np.sort(x[random_choice(np.random.RandomState(),
                                         CDF, n=size)])
            idx = np.searchsorted(_x, vals, side="right")
            _cdfs.append(idx / float(size))
        errs = np.std(_cdfs, axis=0)
    else:
        raise ValueError("`err` can be 'wald' or 'bootstrap'.")

    return cdfs, errs


def weighted_percentile(x, perc, weights=None):
    """
    Calculate the weighted percentile ``perc`` for given data ``x``, so that
    the fraction ``perc`` of the data is smaller equal than
    ``weighted_percentile``.

    Gives the PPF curve (inverse of CDF) when ``perc``is scanned.
    The error on the estimated percentile is approximately the binomial error on
    the fraction of classes using the Wald approximation as stated in
    `https://en.wikipedia.org/wiki/Empirical_distribution_function#Asymptotic_properties`_.

    Inverse function of ``weighted_cdf``.

    Parameters
    ----------
    x : array-like
        Input array with values for which the percentile is calculated.
    perc : float
        Percentile to compute, which must be between 0 and 100 inclusive.
    weights : array-like, optional
        Positive weights for each data point in `x`. If None, weigths are
        all equally set to 1. (default: None)

    Returns
    -------
    percentile : float
        Fraction of weighted data points ``<= perc``.
    err : float
        Estimated uncertainty on the CDF value.

    Note
    ----
    https://stats.stackexchange.com/questions/15891/what-is-the-proper-way-to-estimate-the-cdf-for-a-distribution-from-samples-taken
    https://en.wikipedia.org/wiki/Dvoretzky%E2%80%93Kiefer%E2%80%93Wolfowitz_inequality
    """
    x = np.asarray(x)

    if weights is None:
        weights = np.ones_like(x, dtype=np.float)
    if np.any(weights < 0.) or (np.sum(weights) <= 0):
        raise ValueError("Weights must be positive and add up to > 0.")
    if (perc < 0.) or (perc > 100.):
        raise ValueError("Percentile `perc` must be in [0, 1].")

    perc /= 100.
    # Sort data and add up normalied weight to get empirical CDF
    srt_idx = np.argsort(x)
    weights = weights[srt_idx] / np.sum(weights)
    cdf = np.cumsum(weights)

    # Binomial error on weighted cdf in Wald approximation is used for the
    # percentile because the percentile results from a selection based on cdf.
    mask = (cdf <= perc)
    if np.all(~mask):  # Asked for percentile smaller than smallest CDF value
        return np.amin(x), None
    else:
        percentile = x[srt_idx][mask][-1]
        p = cdf[mask][-1]
        err = np.sqrt(p * (1. - p) * np.sum(weights**2))
        return percentile, err


def cdf_nzeros(x, nzeros, vals, sorted=False):
    """
    Returns the CDF value at value ``vals`` for a dataset with ``x > 0`` and
    ``nzeros`` entries that are zero.

    Parameters
    ----------
    x : array-like
        Data values on which the percentile is calculated.
    nzeros : int
        Number of zero trials.
    vals : float or array-like
        Threshold(s) in x-space to calculate the percentile against.
    sorted : bool, optional
        If ``True`` assume ``x`` is sorted ``x[0] <= ... <= x[-1]``.
        Can save time on large arrays but produces wrong results, if array is
        not really sorted.

    Returns
    -------
    cdf : float
        CDF in ``[0, 1]``, fraction of ``x <= vals``.
    """
    x = np.atleast_1d(x)
    vals = np.atleast_1d(vals)
    if not sorted:
        x = np.sort(x)

    ntot = len(x) + nzeros

    # CDF(x<=val) =  Total fraction of values x <= val + given zero fraction
    frac_zeros = nzeros / float(ntot)
    cdf = np.searchsorted(x, vals, side="right") / float(ntot) + frac_zeros
    return cdf


def percentile_nzeros(x, nzeros, q, sorted=False):
    """
    Returns the percentile ``q`` for a dataset with ``x > 0`` and ``nzeros``
    entries that are zero.

    Alternatively do ``np.percentile(np.r_[np.zeros(nzeros), x], q)``, which
    gives the same result when choosing ``interpolation='lower'``.

    Parameters
    ----------
    x : array-like
        Non-zero values.
    nzeros : int
        Number of zero trials.
    q : float
        Percentile in ``[0, 100]``.
    sorted : bool, optional
        If ``True`` assume ``x`` is sorted ``x[0] <= ... <= x[-1]``.
        Can save time on large arrays but produces wrong results, if array is
        not really sorted.

    Returns
    -------
    percentile : float
        The percentile at level ``q``.

    Example
    -------
    >>> from anapymods3.stats import delta_chi2, percentile_nzeros
    >>> df, eta, perc = 1., 0.01, 99.999
    >>> rvs = delta_chi2.rvs(df, eta, size=int(1e7))
    >>> print("concat: ", np.percentile(rvs, q=perc, interpolation="lower"))
    >>> print("nzeros: ", percentile_nzeros(rvs[rvs > 0.],
                                            int(np.sum(rvs <= 0.)), q=perc))
    >>> print("True  : ", delta_chi2.ppf(perc / 100., df, eta))
    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q) / 100.
    if not sorted:
        x = np.sort(x)

    ntot = len(x) + nzeros
    idx = (q * ntot).astype(int) - nzeros - 1

    percentile = np.zeros_like(q, dtype=np.float)
    m = (idx >= 0)
    percentile[m] = x[idx[m]]

    return percentile


def weighted_percentile_in_bin(x, y, xbins, perc, yweights=None):
    """
    Calculate weighted percentile in each region defined in ``bins``. The bins
    are not used to calculate the median, only to divide the data space in sub
    regions.

    Parameters
    ----------
    x : array-like
        1st input array which is used to split up the data in bins.
    y : array-like
        2nd input array with values for which the percentile per bin is
        calculated.
    xbins : array-like
        Bin edges parting the `x` data array in several regions.
    perc : float
        Percentile to compute per bin, must be between 0 and 100 inclusive.
    yweights : array-like, optional
        Positive weights for each data point in `y`. If None, weigths are
        all equally set to 1. (default: None)

    Returns
    -------
    percentile : float
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    xbins = np.atleast_1d(xbins)
    nbins = len(xbins) - 1
    if nbins < 1:
        raise ValueError("Bins must have at least two entries.")

    if yweights is None:
        yweights = np.ones_like(y, dtype=np.float)

    # Bin the input data x in the given xbins
    idx = np.digitize(x, xbins)

    # For every y in each xbin, get the weighted percentile of these y
    percentile = np.zeros(nbins, dtype=np.float)
    for i in range(nbins - 1):
        mask = (idx == i + 1)
        _y = y[mask]
        _yweights = yweights[mask] / np.sum(yweights[mask])
        percentile[i] = weighted_percentile(_y, perc=perc, weights=_yweights)

    return percentile


def clopper_pearson_interval(k, n, alpha):
    """
    Calculate the Clopper-Pearson exact (but conservative) confidence interval
    with coverage ``alpha`` for an estimation of the succes rate :math:`p` of a
    binomial experiment. The interval is two-sided, so ``(1-alpha) / 2`` of
    probability lies in each tail (``1-alpha`` being the error rate).

    Parameters
    ----------
    k : int or array-like
        Number of successes.
    n : int or array-like
        Number of total trials.
    alpha : float or array-like
        Confidence level of the interval, tail probability or type I error rate
        is then ``1-alpha``. Value(s) must be in ``[0, 1]``.

    Returns
    -------
    lower_lim: array-like, shape (len(alpha), len(k))
        Lower Clopper-Pearson interval boundary for the given ``alpha``s, ``k``s
        and ``n``s.
    upper_lim: array-like, shape (len(alpha), len(k))
        Upper Clopper-Pearson interval boundary for the given ``alpha``s, ``k``s
        and ``n``s.

    Note
    ----
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    alpha = 1. - np.atleast_1d(alpha)[None, :]
    k = np.atleast_1d(k).astype(int)[:, None]
    n = np.atleast_1d(n).astype(int)[:, None]

    if np.any(alpha > 1.) or np.any(alpha < 0.):
        raise ValueError("`alpha` value(s) must be in [0, 1].")
    if k < 0 or n < 0:
        raise ValueError("`n` and `k` value(s) must be >= 0.")

    return (scs.beta.ppf(alpha / 2., k, n - k + 1),
            scs.beta.ppf(1 - alpha / 2, k + 1, n - k))


def percentile_interval(x, q, alpha, sorted=False):
    """
    Calculate the exact (but conservative) confidence interval for a confidence
    level ``alpha`` for the estimated percentile ``q`` given a dataset ``x``.
    The tail probabilty or type I error rate for a given ``alpha`` is then
    ``1-alpha``.

    Parameters
    ----------
    x : array-like
        Dataset values.
    q : float or array-like
        Percentile(s) in ``[0, 1]``.

    Returns
    -------
    lower_lim : array-like
    upper_lim : array-like

    Note
    ----
    https://stats.stackexchange.com/questions/99829/how-to-obtain-a-confidence-interval-for-a-percentile
    """
    raise NotImplementedError("Not implemented yet.")
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)
    alpha = np.atleast_1d(alpha)
    return None
