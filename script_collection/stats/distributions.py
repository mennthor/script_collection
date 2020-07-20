# coding: utf8

"""
Custom distributions, some implementing scipy.stats.rv_continous or rv_discrete.
"""

import numpy as np
from scipy.stats import rv_continuous, chi2


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


class delta_chi2_gen(rv_continuous):
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
        return np.where(x > 0., eta * chi2.pdf(x, df=df), 1. - eta)

    def _logpdf(self, x, df, eta):
        x = np.asarray(x)
        return np.where(x > 0., np.log(eta) + chi2.logpdf(x, df=df),
                        np.log(1. - eta))

    def _cdf(self, x, df, eta):
        x = np.asarray(x)
        return np.where(x > 0., (1. - eta) + eta * chi2.cdf(x, df=df),
                        (1. - eta))

    def _logcdf(self, x, df, eta):
        x = np.asarray(x)
        return np.where(x > 0., np.log(1 - eta + eta * chi2.cdf(x, df)),
                        np.log(1. - eta))

    def _sf(self, x, df, eta):
        # Note: Define sf(0)=0 and not sf(0)=(1-eta) because 0 is inclusive
        x = np.asarray(x)
        return np.where(x > 0., eta * chi2.sf(x, df), 1.)

    def _logsf(self, x, df, eta):
        # Note: Define sf(0)=0 and not sf(0)=(1-eta) because 0 is inclusive
        x = np.asarray(x)
        return np.where(x > 0., np.log(eta) + chi2.logsf(x, df), 0.)

    def _ppf(self, p, df, eta):
        # Derive this from inverting p = delta_chi2.cdf as defined above
        p = np.asarray(p)
        return np.where(p > (1. - eta), chi2.ppf(1 + (p - 1) / eta, df), 0.)

    def _isf(self, p, df, eta):
        # Derive this from inverting p = delta_chi2.sf as defined above
        return np.where(p < eta, chi2.isf(p / eta, df), 0.)

    def fit(self, data, *args, **kwds):
        # Wrapper for chi2 fit, estimating eta and fitting chi2 on data > 0
        data = np.asarray(data)
        eta = float(np.count_nonzero(data)) / len(data)
        df, loc, scale = chi2.fit(data[data > 0.], *args, **kwds)
        return df, eta, loc, scale

    def fit_nzeros(self, data, nzeros, *args, **kwds):
        # Same as `fit` but data has only non-zero trials
        data = np.asarray(data)
        ndata = len(data)
        eta = float(ndata) / (nzeros + ndata)
        df, loc, scale = chi2.fit(data, *args, **kwds)
        return df, eta, loc, scale


delta_chi2 = delta_chi2_gen(name="delta_chi2")
