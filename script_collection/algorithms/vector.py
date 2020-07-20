# coding: utf-8

"""
Vector calculus methods.
"""

import numpy as np


def dist_line_point_3d(sup, dirs, pts):
    """
    Calcaulates the minimum distance from a point to a line in 3D only.

    Parameters
    ----------
    sup : array-like, shape (npts, 3)
        Support vector of the line, each row is a 3D vector.
    dirs : array-like, shape (npts, 3)
        Directional vector of the line, each row is a 3D vector.
    pts : array-like, shape (npts, 3)
        Vector pointing to each point we want the distance to the line from,
        each row is a 3D vector.

    Returns
    -------
    pts : array-like, shape (npts)
        The distance for each point in `pts` to the line.
    """
    def norm(v):
        return np.sqrt(np.sum(v**2, axis=1)).reshape(v.shape[0], 1)

    sup = np.atleast_2d(sup)
    dirs = np.atleast_2d(dirs)
    pts = np.atleast_2d(pts)

    if (sup.shape[0] != dirs.shape[0]) or (sup.shape[0] != pts.shape[0]):
        raise ValueError("All arrays must have the same length.")
    if (sup.shape[1] != 3) or (dirs.shape[1] != 3) or (pts.shape[1] != 3):
        raise ValueError("Vectors in each array must have 3 dimension.")

    return (norm(np.cross(dirs, sup - pts)) / norm(dirs)).ravel()
