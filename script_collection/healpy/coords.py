# coding: utf8

"""
Coordinate helper methods for converting coordinates to and from coordinates in
healpy map convention.
"""

import numpy as np
import healpy as hp


def dec_ra_to_theta_phi(dec, ra):
    """
    Convert equatorial coordinates ``dec, ra`` to healpy coordinates
    ``theta, phi`` using the convention ``dec=pi/2-theta`` and ``phi=ra``.

    Parameters
    ----------
    dec : float or array-like
        Declination in radians.
    ra : float or array-like
        Right-ascension in radians.

    Returns
    -------
    theta : float or array-like
        Healpy polar angle ``theta = pi / 2 - dec``.
    phi : float or array-like
        Healpy azimuthal angle ``phi = ra``.
    """
    return np.pi / 2. - dec, ra


def theta_phi_to_dec_ra(th, phi):
    """
    Convert healpy coordinates ``theta, phi`` to equatorial coordinates
    ``dec, ra`` using the convention ``theta=pi/2-dec`` and ``ra=phi``.

    Parameters
    ----------
    theta : float or array-like
        Healpy polar angle in radians.
    phi : float or array-like
        Healpy azimuthal angle in radians.

    Returns
    -------
    theta : float or array-like
        Equatorial declination ``dec = pi / 2 - theta``.
    phi : float or array-like
        Equatorial right-ascension ``ra = phi``.
    """
    return np.pi / 2. - th, phi


def DecRaToThetaPhi(dec, ra, reverse=False):
    """
    Go from ra, dec to healpy coordinates theta, phi.
    http://stackoverflow.com/questions/29702010/healpy-pix2ang-
    convert-from-healpix-index-to-ra-dec-or-glong-glat
    right to left order: ra = 2pi - phi

    If reverse = True, then RA is from left to right instead from the usual
    right to left order: ra = 2pi - phi

    Note: Same as ``dec_ra_to_theta_phi``
    """
    # In healpy phi goes from right to left with zero in the map center.
    # RA has the same orientation, but starts from the right edge of the map,
    # which is only a matter of drawing.
    # Reverse = True is the same definition as in psLLH allsky-scan and is the
    # standard for equatorial coordinates.
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    if reverse:
        return np.pi / 2. - dec, 2 * np.pi - ra
    else:
        return np.pi / 2. - dec, ra


def ThetaPhiToDecRa(th, phi, reverse=False):
    """
    Go from healpy coordinates to ra, dec.
    http://stackoverflow.com/questions/29702010/healpy-pix2ang-
    convert-from-healpix-index-to-ra-dec-or-glong-glat

    If reverse = True, then RA is from left to right instead from the usual
    right to left order: ra = 2pi - phi

    Note: Same as ``theta_phi_to_dec_ra``
    """
    # In healpy phi goes from right to left with zero in the map center.
    # RA has the same orientation, but starts from the right edge of the map,
    # which is only a matter of drawing.
    # Reverse = True is the same definition as in psLLH allsky-scan and is the
    # standard for equatorial coordinates.
    phi = np.atleast_1d(phi)
    th = np.atleast_1d(th)
    if reverse:
        return np.pi / 2. - th, 2 * np.pi - phi
    else:
        return np.pi / 2. - th, phi


def cos_dist_equ(ra0, dec0, ra1, dec1):
    """
    Cosine of great circle distance in equatorial coordinates between points
    ``(ra0, dec0)`` and ``(ra1, dec1)``.
    Cosine values get clipped at ``[-1, 1]`` to repair float errors that may
    have occurred at the interval edges.

    Note: Use ``angdist_equ`` instead.

    Parameters
    ----------
    ra0, dec0 : floar or array-like
        Equatorial coordinates of first point.
    ra1, dec1 : floar or array-like
        Equatorial coordinates of second point.

    Returns
    -------
    great_circle_dist : float or array-like
        Great circle distance of every ``(ra0, dec0)``, ``(ra1, dec1)`` pair.
    """
    raise DeprecationWarning("Use `angdist_equ` instead.")
    ra0, dec0, ra1, dec1 = [np.atleast_1d(arr)
                            for arr in [ra0, dec0, ra1, dec1]]
    if not len(ra0) == len(dec0) or len(ra1) == len(dec1):
        raise ValueError("Lengths of r0/dec0, ra1/dec1 must match.")

    cos_dist = (np.cos(ra1 - ra0) * np.cos(dec1) * np.cos(dec0)
                + np.sin(dec1) * np.sin(dec0))
    if len(cos_dist) == 1:
        return np.clip(cos_dist, -1., 1.)[0]
    else:
        return np.clip(cos_dist, -1., 1.)


def angdist_equ(ra1, dec1, ra2, dec2):
    """
    Calculate angular distance between all directions ``(ra2, dec1)_i`` and
    all other positions ``(ra2, dec2)_i`` in equatorial coordinates, so
    ``ra`` in ``[0, 2pi]`` and ``dec`` in ``[-pi/2, pi/2]``.

    Parameters
    ----------
    ra1, dec1 : array-like
        Source points in equatorial coordinates in radians.
    ra2, dec2 : array-like
        Target points in equatorial coordinates in radians.

    Returns
    -------
    dist : array, shape (len(ra1), len(ra2))
        Great circle distances in radian.

    Example
    -------
    >>> ra1, dec1 = np.deg2rad([0]), np.deg2rad([0])
    >>> ra2, dec2 = np.deg2rad([0, 90, 180]), np.deg2rad([0, 30, 60])
    >>> print(np.rad2deg(angdist_equ(ra1, dec1, ra2, dec2)))
        array([[   0.,   90.,  120.]])
    """
    dec1 = np.atleast_1d(dec1)[:, None]
    ra1 = np.atleast_1d(ra1)[:, None]
    dec2 = np.atleast_1d(dec2)[None:, ]
    ra2 = np.atleast_1d(ra2)[None:, ]

    cos_dist = (np.cos(ra1 - ra2) * np.cos(dec1) * np.cos(dec2)
                + np.sin(dec1) * np.sin(dec2))
    # Handle possible floating precision errors
    cos_dist = np.clip(cos_dist, -1., 1.)
    return np.arccos(cos_dist)


def angdist_theta_phi(th1, phi1, th2, phi2):
    """
    Calculate angular distance between all directions ``(phi2, th1)_i`` and
    all other positions ``(phi2, th2)_i`` in spherical standard coordinates, so
    ``phi`` in ``[0, 2pi]`` and ``th`` in ``[0, pi]``.

    Parameters
    ----------
    th1, phi1 : array-like
        Source points in healpy coordinates in radians.
    th2, phi2 : array-like
        Target points in healpy coordinates in radians.

    Returns
    -------
    dist : array, shape (len(th1), len(th2))
            Great circle distances in radian.
    """
    th1 = np.atleast_1d(th1)[:, None]
    phi1 = np.atleast_1d(phi1)[:, None]
    th2 = np.atleast_1d(th2)[None:, ]
    phi2 = np.atleast_1d(phi2)[None:, ]

    cos_dist = (np.cos(phi1 - phi2) * np.sin(th1) * np.sin(th2)
                + np.cos(th1) * np.cos(th2))

    # Handle possible floating precision errors
    cos_dist = np.clip(cos_dist, -1, 1)
    return np.arccos(cos_dist)


def angdist(dir1, dir2):
    r"""
    Vectorized calculation of the angular distance between two direction
    angles `dir1` and `dir2`. Each array `dir*` can contain multiple
    directions.

    Directions described by theta, phi are converted to unit vectors first
    and then the angular distance is computed via dot product:

    ..math::

        \mathrm{d} = \arccos(\frac{\vec{v}_1\cdot \vec{v}_2}
                                  {|\vec{v}_1}||\vec{v}_2}|)

    where :math:`\vec{v}_I` is calculated from `dirI`.

    See Wolfram Math World [1]_ or Wikipedia [2]_ for more information on this
    formula.

    Parameters
    ----------
    dir1 : array like
        Array of directions. Shape is `N x 2` and means N directions with 2
        healpy angles (theta, phi) each::

            dir1 = [[dir_1_th, dir_1_phi], ..., [dir_N_th, dir_N_phi]]

    dir2 : array like
        Same shape and meaning as `dir1`.

    Returns
    -------
    distances : array like
        The angular distances in radians. For each input directions in `dir1`
        and `dir2` a distance is calculated.

    Notes
    -----
    .. [1] http://mathworld.wolfram.com/SphericalDistance.html
    .. [2] https://en.wikipedia.org/wiki/Great-circle_distance#Vector_version
    """
    def norm_vec(vec):
        """
        Vectorized normalization of length N array of n-dimensional vectors.

        Parameters
        ----------
        vec : array like
            Shape is `N x n` for N vectors with dimension n each::

                vec = [[vec1_1, ..., vec1_n], ..., [vecN_1, ..., vecN_n]]


        Returns
        -------
        normed : array like
            The normed vectors, with the same shape as `vec`.
        """
        norms = np.linalg.norm(vec, axis=1)
        normed = (vec.T / norms).T
        return normed

    # Sanity
    dir1 = np.atleast_2d(dir1)
    dir2 = np.atleast_2d(dir2)
    if not dir1.shape == dir2.shape:
        raise ValueError("Vector arrays must have the same shape.")

    # Make vectors from directions. First angle is theta, second is phi
    # Vectors are returned normalized by healpy.ang2vec()
    nvec1 = hp.ang2vec(theta=dir1[:, 0], phi=dir1[:, 1])
    nvec2 = hp.ang2vec(theta=dir2[:, 0], phi=dir2[:, 1])

    # Norm input vectors
    # nvec1 = norm_vec(vec1)
    # nvec2 = norm_vec(vec2)

    # Dot product, vectorized calculation
    dot = np.sum(nvec1 * nvec2, axis=1)

    # Clip numerical rounding errors and get angle in radians
    distances = np.arccos(np.clip(dot, -1.0, +1.0))

    return distances


def rotator(ra1, dec1, ra2, dec2, ra3, dec3):
    r"""
    Rotate vectors pointing to directions given by pairs `(ra3, dec3)` by the
    rotation defined by going from `(ra1, dec1)` to `(ra2, dec2)`.

    `ra`, `dec` are per event right-ascension in :math:`[0, 2\pi]` and
    declination in :math:`[-\pi/2, \pi/2]`, both in radians.

    Parameters
    ----------
    ra1, dec1 : array-like, shape (nevts)
        The points we start the rotation at.
    ra2, dec2 : array-like, shape (nevts)
        The points we end the rotation at.
    ra3, dec3 : array-like, shape (nevts)
        The points we actually rotate around the axis defined by the directions
        above.

    Returns
    -------
    ra3t, dec3t : array-like, shape (nevts)
        The rotated directions `(ra3, dec3) -> (ra3t, dec3t)`.

    Notes
    -----
    Using quaternion rotation from [1]_. Was a good way to recap this stuff.
    If you are keen, you can show that this is the same as the rotation
    matrix formalism used in skylabs rotator.

    Alternative ways to express the quaternion conjugate product:

    .. code-block::
       A) ((q0**2 - np.sum(qv * qv, axis=1).reshape(qv.shape[0], 1)) * rv +
            2 * q0 * np.cross(qv, rv) +
            2 * np.sum(qv * rv, axis=1).reshape(len(qv), 1) * qv)

       B) rv + 2 * q0 * np.cross(qv, rv) + 2 * np.cross(qv, np.cross(qv, rv))


    .. [1] http://people.csail.mit.edu/bkph/articles/Quaternions.pdf
    """
    def ra_dec_to_quat(ra, dec):
        r"""
        Convert equatorial coordinates to quaternion representation.

        Parameters
        ----------
        ra, dec : array-like, shape (nevts)
            Per event right-ascension in :math:`[0, 2\pi]` and declination in
            :math:`[-\pi/2, \pi/2]`, both in radians.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            One quaternion per row from each (ra, dec) pair.
        """
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return np.vstack((np.zeros_like(x), x, y, z)).T

    def quat_to_ra_dec(q):
        r"""
        Convert quaternions back to quatorial coordinates.

        Parameters
        ----------
        q : array-like, shape (nevts, 4)
            One quaternion per row to convert to a (ra, dec) pair each.

        Returns
        -------
        ra, dec : array-like, shape (nevts)
            Per event right-ascension in :math:`[0, 2\pi]` and declination in
            :math:`[-\pi/2, \pi/2]`, both in radians.
        """
        nv = norm(q[:, 1:])
        x, y, z = nv[:, 0], nv[:, 1], nv[:, 2]
        dec = np.arcsin(z)
        ra = np.arctan2(y, x)
        ra[ra < 0] += 2. * np.pi
        return ra, dec

    def norm(v):
        """
        Normalize a vector, so that the sum over the squared elements is one.

        Also valid for quaternions.

        Parameters
        ----------
        v : array-like, shape (nevts, ndim)
            One vector per row to normalize

        Returns
        -------
        nv : array-like, shape (nevts, ndim)
            Normed vectors per row.
        """
        norm = np.sqrt(np.sum(v**2, axis=1))
        m = (norm == 0.)
        norm[m] = 1.
        vn = v / norm.reshape(v.shape[0], 1)
        assert np.allclose(np.sum(vn[~m]**2, axis=1), 1.)
        return vn

    def quat_mult(p, q):
        """
        Multiply p * q in exactly this order.

        Parameters
        ----------
        p, q : array-like, shape (nevts, 4)
            Quaternions in each row to multiply.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            Result of the quaternion multiplication. One quaternion per row.
        """
        p0, p1, p2, p3 = p[:, [0]], p[:, [1]], p[:, [2]], p[:, [3]]
        q0, q1, q2, q3 = q[:, [0]], q[:, [1]], q[:, [2]], q[:, [3]]
        # This algebra reflects the similarity to the rotation matrices
        a = q0 * p0 - q1 * p1 - q2 * p2 - q3 * p3
        x = q0 * p1 + q1 * p0 - q2 * p3 + q3 * p2
        y = q0 * p2 + q1 * p3 + q2 * p0 - q3 * p1
        z = q0 * p3 - q1 * p2 + q2 * p1 + q3 * p0

        return np.hstack((a, x, y, z))

    def quat_conj(q):
        """
        Get the conjugate quaternion. This means switched signs of the
        imagenary parts `(i,j,k) -> (-i,-j,-k)`.

        Parameters
        ----------
        q : array-like, shape (nevts, 4)
            One quaternion per row to conjugate.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            Conjugated quaternions. One quaternion per row.
        """
        return np.hstack((q[:, [0]], -q[:, [1]], -q[:, [2]], -q[:, [3]]))

    def get_rot_quat_from_ra_dec(ra1, dec1, ra2, dec2):
        r"""
        Construct quaternion which defines the rotation from a vector
        pointing to `(ra1, dec1)` to another one pointing to `(ra2, dec2)`.

        The rotation quaternion has the rotation angle in it's first
        component and the axis around which is rotated in the last three
        components. The quaternion must be normed :math:`\sum(q_i^2)=1`.

        Parameters
        ----------
        ra1, dec1 : array-like, shape (nevts)
            The points we start the rotation at.
        ra2, dec2 : array-like, shape (nevts)
            The points we end the rotation at.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            One quaternion per row defining the rotation axis and angle for
            each given pair of `(ra1, dec1)`, `(ra2, dec2)`.
        """
        p0 = ra_dec_to_quat(ra1, dec1)
        p1 = ra_dec_to_quat(ra2, dec2)
        # Norm rotation axis for proper quaternion normalization
        ax = norm(np.cross(p0[:, 1:], p1[:, 1:]))

        cos_ang = np.clip((np.cos(ra1 - ra2) * np.cos(dec1) * np.cos(dec2)
                           + np.sin(dec1) * np.sin(dec2)), -1., 1.)

        ang = np.arccos(cos_ang).reshape(cos_ang.shape[0], 1)
        ang /= 2.
        a = np.cos(ang)
        ax = ax * np.sin(ang)
        # Normed because: sin^2 + cos^2 * vec(ax)^2 = sin^2 + cos^2 = 1
        return np.hstack((a, ax[:, [0]], ax[:, [1]], ax[:, [2]]))

    ra1, dec1, ra2, dec2, ra3, dec3 = map(np.atleast_1d,
                                          [ra1, dec1, ra2, dec2, ra3, dec3])
    assert(len(ra1) == len(dec1) == len(ra2) == len(dec2)
           == len(ra3) == len(dec3))

    # Convert (ra3, dec3) to imaginary quaternion -> (0, vec(ra, dec))
    q3 = ra_dec_to_quat(ra3, dec3)

    # Make rotation quaternion: (angle, vec(rot_axis)
    q_rot = get_rot_quat_from_ra_dec(ra1, dec1, ra2, dec2)

    # Rotate by multiplying q3' = q_rot * q3 * q_rot_conj
    q3t = quat_mult(q_rot, quat_mult(q3, quat_conj(q_rot)))
    # Rotations preserves vectors, so imaganery part stays zero
    assert np.allclose(q3t[:, 0], 0.)

    # And transform back to (ra3', dec3')
    return quat_to_ra_dec(q3t)
