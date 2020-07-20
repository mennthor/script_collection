# coding: utf-8

"""
Mathematical stuff.
"""

import numpy as np
from itertools import permutations


def wrap_angle(angles, wrap_angle=2 * np.pi):
    r"""
    Wraps angles :math:`\alpha` to range
    :math:`\Phi - 2\pi \leq \alpha < \Phi`, where :math:`\Phi` is the
    `wrap_angle`.

    Parameters
    ----------
    angles : array-like
        Angles in radian.
    wrap_angle : float, optional
        Defines the upper border of the wrapping interval. Default is
        :math:`2\pi`, so angles are wrapped to :math`[0, 2\pi]`.

    Returns
    -------
    wrapped : array-like
        Wrappend angles :math:`\alpha` in radian in range
        :math:`\Phi - 2\pi <= \alpha < \Phi`

    Notes
    -----
    This method is taken 1:1 from `astropy.coordinates.Angle.wrap_at` [1].

    The method seems to be a clever way to avoid handling cases I guess.
    Naively we would see which angles are inside, less or greater the given
    range. Then we would use the modulo to decide how many intervals they're
    off and then add that many times the interval size if we were less or
    subtract if we were greater than the interval.

    In this form here we offset all angles by wrap_angle initially instead.
    This way the modulo is offset by negative one which automatically
    handles both cases correctly, when we subtract (360 - wrap_angle) after
    the mod to correct for thje initial offset.

    .. [1] http://docs.astropy.org/en/stable/api/astropy.coordinates.Angle.html#astropy.coordinates.Angle.wrap_at # noqa
    """
    return (np.mod(angles - wrap_angle, 2 * np.pi) - (2 * np.pi - wrap_angle))


def get_perms(ndim):
    """
    Returns all locations of neighbours of a nD hypercube.

    - 1D has  2 = 3^1 - 1
    - 2D has  8 = 3^2 - 1
    - 3D has 26 = 3^3 - 1

    and so on.

    Returned is a list of tuples. Each tuple is holding -1, 0, 1 and specifies
    the location of the neighbour hypercube relativ to the center.
    Eg. (-1, 1, 0) means the neighbour at the left in x, at the right in y and
    in front of z coordinate.

    In 2D it gets simpler to understand:

        +---------+---------+---------+
        |         |         |         |
        | (-1,+1) | ( 0, 1) | (+1,+1) |
        |         |         |         |
        +---------+---------+---------+
        |         |         |         |
        | (-1, 0) | ( 0, 0) | (+1, 0) |
        |         |         |         |
        +---------+---------+---------+
        |         |         |         |
        | (-1,-1) | ( 0,-1) | (-1,-1) |
        |         |         |         |
        +---------+---------+---------+

    Parameters
    ----------
    ndim : int
        How many dimensions we have.

    Returns
    -------
    perms : array-like, shape (3**ndim - 1, ndim)
        [ perm(0), perm(1), ..., perm(3**ndim - 1) ] as explained above.
    """
    coords = np.array([-1, 0, 1])

    # Expand and get all permutations
    perms = []
    for perm in permutations(np.repeat(coords, ndim), ndim):
        perms.append(perm)

    # Reduce to unique set
    perms = set(perms)

    # Put into array
    perm_arr = np.empty((len(perms), ndim), dtype=int)
    for i, perm in enumerate(perms):
        perm_arr[i] = np.array(perm, dtype=int)

    return perm_arr


def ln_factorial(k):
    """
    Calculates ``ln(k!)`` for each ``k_i`` in ``k``.

    Parameters
    ----------
    k : array-like
        Integers to get the log factorials for.

    Returns
    -------
    ln_fac : array-like
        Log factorial for each element in ``k``.
    """
    k = np.atleast_1d(k)
    res = np.zeros(len(k), dtype=float)
    for i, ki in enumerate(k):
        if ki > 0:
            res[i] = np.sum(np.log(np.arange(1, ki + 1)))
    return res
