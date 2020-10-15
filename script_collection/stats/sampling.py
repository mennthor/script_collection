# coding: utf-8

"""
Contains random sampling methods.
"""
import math
import numpy as np
import scipy.optimize as sco
import scipy.stats as scs
from sklearn.utils import check_random_state
import emcee


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


def sample_lognorm(lo, hi, cutoff=None, size=None, rndstate=None):
    """
    Samples log uniformly from interval [hi, lo]. Also considers intervals
    spanning zero or being in the negative range. If the interval crosses zero,
    a cutoff must be provided to split the sampling into two intervals with
    [lo, cutoff] and [cutoff, hi].

    Parameters
    ----------
    lo, hi : float
        Upper and lower border of the sampling range. Must not be inside
        `[-cutoff, cutoff]`.
    cutoff : float or None, optional
        Cutoff towards zero if the given range includes zero. Is not used if
        the range is only in negative or positive space. (default: None)
    size : int or None, optional
        Size of the sample, If `None`, a single number is returned, else a
        ndarray. (default: `None`)
    rndstate : `numpy.random.RandomState`, int or None, optional
        A random state or seed put into `numpy.random.RandomState`.
        (default: `None`)

    Returns
    -------
    rvs : ndarray or float
        Log-uniformly sampled numbers from the given range. If `size` was
        `None`, a single number is returned, else a ndarray.
    """
    rnge = hi - lo
    if not rnge > 0:
        raise ValueError("Must provide range 'lo' < 'hi'.")
    try:
        math.log(abs(lo))
        math.log(abs(hi))
    except ValueError:
        raise ValueError("Interval bounds must not be 0.")

    rndgen = np.random.RandomState(rndstate)

    # 3 cases: Only <0, only >0, including zero
    # Only >=0, sample normally in positive log space
    if hi > 0. and not lo < 0.:
        rvs = scs.loguniform.rvs(
            lo, hi, size=size, random_state=rndgen)
    # Only <=0, sample abs range and add negative sign
    elif not hi > 0. and lo < 0.:
        rvs = -scs.loguniform.rvs(
            abs(hi), abs(lo), size=size, random_state=rndgen)
    # Crossing zero, sample negative portion separately and add sign
    else:
        if cutoff is None:
            raise ValueError(
                "Porived range includes zero, but no cutoff is given.")
        if not cutoff > 0:
            raise ValueError("Zero log 'cutoff' must be > 0.")
        if abs(lo) < cutoff or abs(hi) < cutoff:
            raise ValueError("Provided range ends inside cutoff.")

        p = hi / rnge  # Relative size of spinup parameter range
        _ntrials = 1 if size is None else size
        _n_pos = scs.binom.rvs(_ntrials, p, random_state=rndgen)
        rvs = np.zeros(_ntrials, dtype=float)
        rvs[:_n_pos] = scs.loguniform.rvs(
            cutoff, hi, size=_n_pos, random_state=rndgen)
        rvs[_n_pos:] = -scs.loguniform.rvs(
            cutoff, abs(lo), size=_ntrials - _n_pos, random_state=rndgen)
        rndgen.shuffle(rvs)

    return rvs[0] if size is None else rvs


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


def create_binned_poisson_llh_sampler(data, bins, components, log_prior, seeds,
                                      nwalkers, comp_kwargs={}, **emcee_kwargs):
    """
    Creates an `emcee.EnsembleSampler` with a multi-component, binned Poisson
    LLH being ready to sample.

    Parameters
    ----------
    data : array-like
        Data sample to histogram.
    bins : array-like or int
        Given to `numpy.histogram`. Each bin is considered in the fitting
        procedure.
    components : dict
        Dictionary of callables. Each callable for a given `name` is called as
        `components[name](params, bins, **comp_kwargs[name])`, where the correct
        parameter array is inferred from `seeds[name]` and is an array
        describing a single point in the parameter space of that component.
        Each callable is expected to return the expectation value in `[0, inf]`
        in each bin. Additional arguments are passed via `comp_kwargs[name]`.
    log_prior : callable
        Log of thr prior distriution, is added to the Poisson logLLH. Is called
        as `log_prior(params, **comp_kwargs)`, where `params` is a dict with the
        parameter arrays as values for each component name and `comp_kwargs` is
        the full dict of extra keyword arguments which can optionally be given
        to this method.
        Make sure the resulting posterior LLH is properly normalized up to a
        constant value. The prior can also be used for constraints and
        boundaries by returning `-np.inf` for a forbidden parameter range or
        combination.
    nwalkers : int
        The number of walkers in the ensemble.
    seeds : dict
        Initial parameter values for each component. Each seed must be given as
        a 2D array with shape `nwalkers, ndim` containing one seed point per
        `emcee` walker for the selected component.
    comp_kwargs : dict of dicts, optional
        Additional keyword arguments for each component and prior callable.
        (default: `{}`)
    **emcee_kwargs :
        Additional keyword arguments passed directly to the underlying
        `emcee.EnsembleSampler`.

    Returns
    -------
    TODO
    """
    # Check input parameters
    if nwalkers < 1:
        raise ValueError("Need at least a single walker.")

    for k in ["ndim", "log_prob_fn", "args", "kwargs"]:
        if k in emcee_kwargs:
            raise KeyError("'{}' is an emcee.EnsembleSample setting that is "
                           "set internally and can't be overriden.")

    # Construct the proper parameter arrays for the sampling
    x0 = []
    ndim = {}
    slices = {}
    _offset = 0
    for name in components.keys():
        if name == "_GLOBAL_":
            raise ValueError("A component cannot be named '_GLOBAL_'. This key "
                             "is reserved for a global prior component.")
        if name not in seeds:
            raise ValueError(
                "Missing initial values for component '{}'".format(name))
        _seeds = np.atleast_2d(seeds[name])
        if not _seeds.shape[0] == nwalkers:
            raise ValueError(
                "Initial values for component '{}' does not contain as many "
                "seed points as requested walkers.")
        ndim[name] = _seeds.shape[1]
        x0.append(seeds[name])
        # Slice selects the matching params from the merged param array later
        slices[name] = slice(_offset, _offset + ndim[name])
        _offset += ndim[name]
        # Fill comp_kwargs with empty dicts to avoid checks in the LLH later
        if name not in comp_kwargs:
            comp_kwargs[name] = {}

    # All seeds concatendated to a single point per walker in the correct order
    x0 = np.hstack(x0)

    # The data histogram stays fixed during the whole procedure
    hist, bins = np.histogram(data, bins)

    # This avoids -inf values when an expectation is 0. Clip it to lowest value
    CLIP = -np.floor(np.log10(abs(np.finfo(float).min))) - 1
    while np.isinf(scs.poisson.logpmf(1, 10**(CLIP))):
        CLIP += 1
    CLIP = 10**(CLIP + 1)

    # This is the logLLH that gets sampled
    def log_llh(params, slices, hist, bins, components, log_prior,
                comp_kwargs, CLIP):
        # Evaluate prior first, because it contains boundary checks. If the
        # prior is already -inf, don't bother evaluating the actual function
        p_dict = {k: params[s] for k, s in slices.items()}
        logprior = log_prior(p_dict, **comp_kwargs)
        if np.isinf(logprior):
            return -np.inf  # Return early, unreachable parameter point

        # Distribute proper params to each component. Each component is required
        # to return an array of expectation values >=0, one for each bin.
        mus = np.zeros(len(bins) - 1, dtype=float) + CLIP
        for name, func in components.items():
            _p = params[slices[name]]

            # Compute expectations
            mus_comp = func(_p, bins, **comp_kwargs[name])
            if np.any(mus_comp < 0):
                raise ValueError(
                    "Component '{}' returned expectation values <0 for "
                    "parameters {}.".format(
                        name, ", ".join(["{:.3g}".format(pi) for pi in _p])))
            mus += mus_comp

        # Compute Poisson logLLH and add log prior for the final log posterior
        logllh = np.sum(scs.poisson.logpmf(hist, mus))
        return logllh + logprior

    # Create the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers=nwalkers,
        ndim=sum(ndim.values()),
        log_prob_fn=log_llh,
        args=(slices, hist, bins, components, log_prior, comp_kwargs, CLIP),
    )

    # The slices allow matching the resulting parameter arrays to the components
    return sampler, x0, slices
