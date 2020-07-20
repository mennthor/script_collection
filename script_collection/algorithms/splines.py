# coding: utf-8

"""
scipy spline related methods.
"""

import numpy as np
import scipy.interpolate as sci


class spl_normed(object):
    """
    Simple wrapper to make and handle a normalized UnivariateSpline.

    The given spline is normalized so that integral over ``[lo, hi]`` is
    ``norm``. There might be a better way by directly inheriting from
    UnivariateSpline, but this is OK, if we don't need the full feature set.

    Parameters
    ----------
    spl : scipy.interpolate.UnivariateSpline instance
        A spline object that shall be normlaized.
    norm : float
        The value the new spline's integral should have over ``lo, hi``.
    lo, hi : float
        Lower and upper integration borders over which the integral should be
        ``norm``.
    """
    def __init__(self, spl, norm, lo, hi):
        self._spl = spl
        if spl.integral(a=lo, b=hi) == 0:
            raise ValueError("Given spline has integral 0, can't scale it.")
        self._scale = norm / spl.integral(a=lo, b=hi)

    def __call__(self, x, nu=0, ext=None):
        return self._scale * self._spl(x, nu, ext)

    def antiderivative(self, n=1):
        return self._scale * self._spl.antiderivative(n)

    def derivative(self, n=1):
        return self._scale * self._spl.derivative(n)

    def derivatives(self, x):
        return self._scale * self._spl.derivatives(x)

    def get_coeffs(self):
        return self._scale * self._spl.get_coeffs()

    def get_knots(self):
        return self._spl.get_knots()

    def get_residual(self):
        raise NotImplementedError("Don't knwo how to do this.")

    def integral(self, a, b):
        return self._scale * self._spl.integral(a, b)

    def roots(self, ):
        return self._spl.roots()

    def set_smoothing_factor(self, s):
        raise NotImplementedError("Don't knwo how to do this.")


def make_spl_edges(vals, bins, w=None):
    """
    Make nicely behaved edge conditions for a spline fit to a histogram.

    Edge values are modeled to be a sane and constraint value between the next
    and next to next bin to fix the spline at the histogram edges.

    Parameters
    ----------
    vals : array-like
        Histogram values.
    bins : array-like
        Bin edges used to create the histogram.
    w : array-like or None, optional
        Weights that may be used in the spline creation. Gets the correct shape
        in this method if not ``None``. (default: ``None``)

    Returns
    -------
    vals : array-like
        Same as before, but with added edge values.
    pts : array-like
        Points for every value. Outermost points are the outermost bin edges,
        inner points are bin centers.
    w : array-like or None
        ``None`` if previously ``None`` or with new shape to fit ``vals``.
    """
    vals = np.atleast_1d(vals)
    bins = np.atleast_1d(bins)
    if len(vals) != len(bins) - 1:
        raise ValueError("Bin egdes must have length `len(vals) + 1`.")
    if w is not None:
        w = np.atleast_1d(w)
        if len(w) != len(vals):
            raise ValueError("Weights must have same length as vals")
        w = np.concatenate((w[[0]], w, w[[-1]]))

    # Model outermost bin edges to avoid uncontrolled behaviour at the edges
    if len(vals) > 2:
        # Subtract mean of 1st and 2nd bins from 1st to use as height 0
        val_l = (3. * vals[0] - vals[1]) / 2.
        # The same for the right edge
        val_r = (3. * vals[-1] - vals[-2]) / 2.
    else:  # Just repeat if we have only 2 bins
        val_l = vals[0]
        val_r = vals[-1]

    vals = np.concatenate(([val_l], vals, [val_r]))
    mids = 0.5 * (bins[:-1] + bins[1:])
    pts = np.concatenate((bins[[0]], mids, bins[[-1]]))
    return vals, pts, w


def fit_spl_to_hist(h, bins, stddev=None):
    """
    Takes histogram values, their standard deviations and bin edges and returns
    a spline fit through the bin centers.

    Parameters
    ----------
    h : array-like
        Histogram values.
    bins : array-like
        Bin edges used to create the histogram.
    stddev : array-like or None, optional
        Used as weights to create a smoothing spline. If ``None`` the spline is
        interpolating. (default: ``None``)

    Returns
    -------
    spl : scipy.interpolate.UnivariateSpline instance
        Spline fitted to the histogram.
    norm : float
        Integral of the spline between the outermost bin edges.
    vals : array-like
        y values used to create the spline.
    pts : array-like
        x values used to create the spline
    """
    h, bins = map(np.atleast_1d, [h, bins])
    if len(h) != len(bins) - 1:
        raise ValueError("Bin edges must have length `len(h) + 1`.")
    if stddev is not None:
        stddev = np.atleast_1d(stddev)
        if len(h) != len(stddev):
            raise ValueError("Length of errors and histogram muste be equal.")
        s = len(h)
        if np.any(stddev <= 0):
            raise ValueError("Given stddev has unallowed entries <= 0.")
        w = 1. / stddev
        assert len(w) == s == len(h) == len(stddev)
    else:
        s = 0
        w = None

    vals, pts, w = make_spl_edges(h, bins, w=w)
    spl = sci.UnivariateSpline(pts, vals, s=s, w=w, ext="raise")
    norm = spl.integral(a=bins[0], b=bins[-1])
    return spl, norm, vals, pts
