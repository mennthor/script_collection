# coding: utf8

"""
Custom distributions, some implementing scipy.stats.rv_continous or rv_discrete.
"""

import numpy as np
import scipy.stats as scs


def power_law_norm(gamma, xlow, xhig):
    r"""
    Calculate the power law pdf normalization with the given parameters, so the
    area under the pdf is 1 when normalized with this normalization constant.
    Is used in `power_law_pdf` function. The :math:`\gamma=1` case is taken
    care of.

    Parameters
    ----------
    gamma : float
        Power law index.
    xlow, xhig : float
        Border of the pdf, needed for proper normalization.
    n : int
        Number of events to be sampled.

    Returns
    -------
    norm : float
        Normalization constant for a power law with the given parameters.
    """
    if gamma == 1:
        return np.log(xhig) - np.log(xlow)
    else:
        return (xhig**(1. - gamma) - xlow**(1. - gamma)) / (1. - gamma)


def power_law_pdf(x, gamma, xlow, xhig):
    r"""
    Calculate the pdf value for a power law with the given values. The power
    law pdf is given by:

    .. math::
       \mathrm{pdf}(\gamma) = x^{-\gamma} / \mathrm{norm}

    where norm ensures an area under curve of one. Positive spectral index
    gamma means a falling spectrum.

    Note: When :math:`\gamma=1` the integral is

    .. math::
       \int 1/x \mathrm{d}x = ln(x) + c

    This case is also handled. The pdf is implemented completely analytic, so
    no numeric integration is used to normalize.
    But it can be easily checked that area under curve is 1 using::

        area, err = scipy.integrate.quad(power_law_pdf, Elow, Ehig)
        print(area)  # Should be 1.0

    Parameters
    ----------
    gamma : float
        Power law index.
    xlow, xhig : float
        Border of the pdf, needed for proper normalization.
    n : int
        Number of events to be sampled.

    Returns
    -------
    pdf : float array
        Array with the values of the power law pdf at given x values: pdf(x).
    """
    norm = power_law_norm(gamma, xlow, xhig)
    return x**(-gamma) / norm


class delta_chi2_gen(scs.rv_continuous):
    r"""
    Class for a probability denstiy function modelled by using a:math`\chi^2`
    distribution for :math:`x > 0` and a constant fraction :math:`1 - \eta`
    of zero trials for :math`x = 0` (like a delta peak at 0).

    Notes
    -----
    The probability density function for `delta_chi2` is:

    .. math::

      \text{PDF}(x|\text{df}, \eta) =
          \begin{cases}
              (1-\eta)                &\text{for } x=0 \\
              \eta\chi^2_\text{df}(x) &\text{for } x>0 \\
          \end{cases}

    `delta_chi2` takes ``df`` and ``eta``as a shape parameter, where ``df`` is
    the standard :math:`\chi^2_\text{df}` degrees of freedom parameter and
    ``1-eta`` is the fraction of the contribution of the delta function at zero.
    """
    def _rvs(self, df, eta):
        # Determine fraction of zeros by drawing from binomial with p=eta
        s = self._size if not len(self._size) == 0 else 1
        nzeros = self._random_state.binomial(n=s, p=(1. - eta), size=None)
        # If None, len of size is 0 and single scalar rvs requested
        if len(self._size) == 0:
            if nzeros == 1:
                return 0.
            else:
                return self._random_state.chisquare(df, size=None)
        # If no zeros or no chi2 is drawn for this trial, only draw one type
        if nzeros == self._size:
            return np.zeros(nzeros, dtype=np.float)
        if nzeros == 0:
            return self._random_state.chisquare(df, size=self._size)
        # All other cases: Draw, concatenate and shuffle to simulate a per
        # random number Bernoulli process with p=eta
        out = np.r_[np.zeros(nzeros, dtype=np.float),
                    self._random_state.chisquare(df,
                                                 size=(self._size - nzeros))]
        self._random_state.shuffle(out)
        return out

    def _pdf(self, x, df, eta):
        x = np.asarray(x)
        return np.where(x > 0., eta * scs.chi2.pdf(x, df=df), 1. - eta)

    def _logpdf(self, x, df, eta):
        x = np.asarray(x)
        return np.where(x > 0., np.log(eta) + scs.chi2.logpdf(x, df=df),
                        np.log(1. - eta))

    def _cdf(self, x, df, eta):
        x = np.asarray(x)
        return np.where(x > 0., (1. - eta) + eta * scs.chi2.cdf(x, df=df),
                        (1. - eta))

    def _logcdf(self, x, df, eta):
        x = np.asarray(x)
        return np.where(x > 0., np.log(1 - eta + eta * scs.chi2.cdf(x, df)),
                        np.log(1. - eta))

    def _sf(self, x, df, eta):
        # Note: Define sf(0)=0 and not sf(0)=(1-eta) because 0 is inclusive
        x = np.asarray(x)
        return np.where(x > 0., eta * scs.chi2.sf(x, df), 1.)

    def _logsf(self, x, df, eta):
        # Note: Define sf(0)=0 and not sf(0)=(1-eta) because 0 is inclusive
        x = np.asarray(x)
        return np.where(x > 0., np.log(eta) + scs.chi2.logsf(x, df), 0.)

    def _ppf(self, p, df, eta):
        # Derive this from inverting p = delta_chi2.cdf as defined above
        p = np.asarray(p)
        return np.where(p > (1. - eta), scs.chi2.ppf(1 + (p - 1) / eta, df), 0.)

    def _isf(self, p, df, eta):
        # Derive this from inverting p = delta_chi2.sf as defined above
        return np.where(p < eta, scs.chi2.isf(p / eta, df), 0.)

    def fit(self, data, *args, **kwds):
        # Wrapper for chi2 fit, estimating eta and fitting chi2 on data > 0
        data = np.asarray(data)
        eta = float(np.count_nonzero(data)) / len(data)
        df, loc, scale = scs.chi2.fit(data[data > 0.], *args, **kwds)
        return df, eta, loc, scale

    def fit_nzeros(self, data, nzeros, *args, **kwds):
        # Same as `fit` but data has only non-zero trials
        data = np.asarray(data)
        ndata = len(data)
        eta = float(ndata) / (nzeros + ndata)
        df, loc, scale = scs.chi2.fit(data, *args, **kwds)
        return df, eta, loc, scale


delta_chi2 = delta_chi2_gen(name="delta_chi2")


def dist_poisson(counts, alphas, **kwargs):
    """
    Poisson central intervals compliant with `add_residual_plot`.

    Parameters
    ----------
    counts : array-like
        The actual data counts.
    alphas : array-like
        Confidence levels for the central intervals that shall be computed.
    kwargs :
        'mus' : array-like
            Expectation values for the poisson distribution.
            Must have same length as `counts`.

    Returns
    -------
    alpha_cnts : array-like
        CDF value for each count.
    alpha_expect : array-like
        CDF value for each expectation value for the corresponding distribution
        used per count (used for scaling in `add_residual_plot`).
    alpha_intervals : list
        List of low and high interval CDF values for the corresponding
        distribution used per count. For each alpha, the list contains a tuple
        `(lo, hi)` with `lo`, `hi` being arrays of CDF values with length
        `len(counts)`.
    """
    mus = kwargs["mus"]
    # Get alphas for counts and for the expectation values
    alpha_cnts = scs.poisson.cdf(counts, mus)
    alpha_expect = scs.poisson.cdf(mus, mus)
    # Create intervals for each given alpha
    alphas = np.unique(alphas)
    if not np.all(np.logical_and(alphas > 0, alphas < 1)):
        raise ValueError("Interval alphas must be in (0, 1).")
    alpha_intervals = []
    for alpha in alphas:
        lo, hi = scs.poisson.interval(alpha, mus)
        # Because the Poisson distribution is discrete, we have to remove
        # the lower border of the intervall manually, otherwise too much
        # probability is included (the intervals are inclusive in both
        # directions, but the CDF is right inclusive, so for low we need to
        # remove the endpoint).
        alpha_lo = np.clip(
            scs.poisson.cdf(lo, mus) - scs.poisson.pmf(lo, mus), 0, 1)
        alpha_hi = scs.poisson.cdf(hi, mus)
        alpha_intervals.append((alpha_lo, alpha_hi))

    return alpha_cnts, alpha_expect, alpha_intervals


def dist_norm(counts, alphas, **kwargs):
    """
    Gaussian central intervals compliant with `add_residual_plot`.
    Using sqrt(N) residuals per bin.

    Parameters
    ----------
    counts : array-like
        The actual data counts.
    alphas : array-like
        Confidence levels for the central intervals that shall be computed.
    kwargs :
        'mean' : array-like
            Expectation values for the poisson distribution.
            Must have same length as `counts`.

    Returns
    -------
    alpha_cnts : array-like
        CDF value for each count.
    alpha_expect : array-like
        CDF value for each expectation value for the corresponding distribution
        used per count (used for scaling in `add_residual_plot`).
    alpha_intervals : list
        List of low and high interval CDF values for the corresponding
        distribution used per count. For each alpha, the list contains a tuple
        `(lo, hi)` with `lo`, `hi` being arrays of CDF values with length
        `len(counts)`.
    """
    mean = kwargs["mean"]
    stddev = np.sqrt(mean)
    valid = (stddev > 0.)
    # Get alphas for counts and for the expectation values
    alpha_cnts = np.zeros(len(counts), dtype=float)
    alpha_expect = np.zeros_like(alpha_cnts)
    alpha_cnts[valid] = scs.norm.cdf(counts[valid], mean[valid], stddev[valid])
    alpha_expect[valid] = scs.norm.cdf(mean[valid], mean[valid], stddev[valid])
    # Create intervals for each given alpha
    alphas = np.unique(alphas)
    if not np.all(np.logical_and(alphas > 0, alphas < 1)):
        raise ValueError("Interval alphas must be in (0, 1).")
    alpha_intervals = []
    # Below is very explicit. Gaussian is always symmetric, so it would be the
    # same result for: lo=(1-alpha)/2, hi=(1+alpha)/2=1-lo
    for alpha in alphas:
        lo, hi = np.zeros_like(alpha_cnts), np.zeros_like(alpha_cnts)
        lo[valid], hi[valid] = scs.norm.interval(
            alpha, mean[valid], stddev[valid])
        lo[valid] = scs.norm.cdf(lo[valid], mean[valid], stddev[valid])
        hi[valid] = scs.norm.cdf(hi[valid], mean[valid], stddev[valid])
        alpha_intervals.append((lo, hi))

    return alpha_cnts, alpha_expect, alpha_intervals


def stats_binned(pars, bins, dist, fpars=None):
    """
    Returns the expected number of counts per bin for a
    `scipy.stats.rv_continuous` distribution by integrating over the PDF in each
    bin, normalizing by the bin size and multiplying with a global
    normalization.

    Note: No checks on the parameters are done (because the actual distribution
    is not known). This has to be done beforehand or potential erros handled
    separately.

    Parameters
    ----------
    pars : array-like
        Parameters inserted into `fpars` where fpars is `NaN` or taken
        completely as `fpars` if `fpars` is `None`.
    bins : array-like
        The explicit bin edges for which to return the expected counts.
    dist : scipy.stats.rv_continuous
        A continuous random variable distribtuion for which the expectation is
        calculated by using the CDF to compute the integral in each bin.
    fpars : array-like or None, optional
        Array of parameter values. Is used as
        `fpars[0] * dist.cdf(bins, *fpars[1:])`.
        If `None`, fpars is substitued by `pars`.
        Else, its length must match the expected number of parameters of `dist`
        plus 1 for the total normalization.
        Each `NaN` entry in `fpars` is filled with the corresponding value in
        `pars` keeping the ordering intact. See example for why this is made so
        complicated. (default: `None`)

    Example
    -------
    The usage of `pars` and `fpars` seems a bit complicated but it allows to fix
    a varying amount of parameters for different `dist` methods.
    For example, we want to use `scipy.stats.chi2`, which receives
    `df, loc, scale` parameters, `df` and `scale` should be fixed to 2 and 1.5
    respectively. We can then call this method with
    ```
    pars = np.array([norm, loc])  # bins and pars values set elsewhere
    fpars = np.array([np.nan, 2., np.nan, 1.5])
    stats_binned(pars, bins, scipy.stats.chi2, fpars)
    ```
    If we want to use a gaussian instead, with no fixed parameters, then we use
    ```
    pars = np.array([norm, loc, scale])  # bins and pars values set elsewhere
    fpars = None  # Or equivalently: np.array(len(pars) * [np.nan])
    stats_binned(pars, bins, scipy.stats.norm, fpars)
    ```
    This is of course only useful when `pars` are intended to be varied and
    `fpars` should stay fixed, eg. during an optimization step.
    """
    if fpars is None:
        fpars = pars
    else:
        fpars[np.isnan(fpars)] = pars
    norm, args = fpars[0], fpars[1:]
    integral = dist.cdf(bins[1:], *args) - dist.cdf(bins[:-1], *args)
    return norm * integral / np.diff(bins)
