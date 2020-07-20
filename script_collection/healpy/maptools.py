# coding: utf8

"""
Tools to manipulate or generate healpy maps.
"""

import numpy as np
import healpy as hp
import scipy.optimize as sco

from ..algorithms.math import wrap_angle
from .coords import angdist, DecRaToThetaPhi


def smooth_and_norm_healpy_map(logl_map, smooth_sigma=None):
    """
    Takes a lnLLH map, converts it to normal space, applies gaussian smoothing
    and normalizes it, so that the integral over the unit sphere is 1.

    Parameters
    ----------
    logl_map : array-like
        healpy map array with logLLH values.
    smooth_sigma : float or None, optional
        Width in sigma of gaussian smoothing kernel, must be ``>0.``.
        (default: None)

    Returns
    -------
    pdf_map : array-like
        Smoothed and normalized spatial PDF map.
    """
    if smooth_sigma < 0.:
        raise ValueError("`smooth_sigma` can be in range [0, *].")

    # Normalize to sane values in [*, 0] for conversion llh = exp(logllh)
    pdf_map = np.exp(logl_map - np.amax(logl_map))

    # Smooth with a gaussian kernel
    pdf_map = hp.smoothing(map_in=pdf_map, sigma=smooth_sigma, verbose=False)
    # Healpy smoothing may produce numerical erros, so fix them after smoothing
    pdf_map[pdf_map < 0.] = 0.

    # Normalize to PDF, integral is the sum over discrete pixels here
    NSIDE = hp.get_nside(logl_map)
    dA = hp.nside2pixarea(NSIDE)
    norm = dA * np.sum(pdf_map)
    if norm > 0.:
        pdf_map = pdf_map / norm
        assert np.isclose(np.sum(pdf_map) * dA, 1.)
    else:
        print("  !! Map norm is < 0. Returning unnormed map instead !!")

    return pdf_map


def wrap_theta_phi_range(th, phi):
    """
    Shift over-/underflow of theta, phi angles back to their valid ranges.

    Parameters
    ----------
    th, phi : array-like
        Theta and phi angles in radians. Ranges are `th` in :math`[0, \\pi]`
        and `phi`in :math`[0, 2\\pi]`.

    Returns
    -------
    th, phi : array-like
        Same angles as before, but those outside the ranges are shifted back
        to the sphere correctly.

    Notes
    -----
    Phi is easy to do, because it's periodic on 2pi on the sphere, so we can
    simply remap in the same fashion `astropy.coordinates.Angle.wrap_at` does.

    Theta is more difficult because it's not peridioc. If theta runs outside
    it's range, it's going over the poles.
    Thus the theta angle is decresing again, but phi gets flipped by 180°.
    So theta is treated as 2pi periodic first, so going a whole round over the
    poles we end up where we started.
    Then we mirror angles greater pi so the range (pi, 2pi] is mapped back to
    the range (pi, 0].
    Simultaniously phi is mirrored 180° coming down the other side of the pole.
    Then phi is shifted to it's valid range in [0, 2pi].
    """
    # Start with a correct phi angle
    phi = wrap_angle(phi, np.deg2rad(360))

    # First pretend that theta is peridic in 360°. So going one round over the
    # poles gets us to where we started
    th = wrap_angle(th, np.deg2rad(360))

    # Now mirror the thetas >180° to run up the sphere again.
    # Eg. th = 210° is truely a theta of 150°, th = 300° is 60°, etc.
    m = th > np.deg2rad(180)
    th[m] = 2 * np.deg2rad(180) - th[m]

    # For the mirrored thetas we move the phis 180° around the sphere,
    # which is all that happens when crossing the poles on a sphere
    phi[m] = phi[m] + np.deg2rad(180)

    # Now put the phis which where transformed to vals > 360° back to [0°, 360°]
    phi = wrap_angle(phi, np.deg2rad(360))

    return th, phi


def add_weighted_maps(m, w):
    """
    Add weighted healpy maps. Lenght of maps m and weights w must match.
    Maps have shape (nmaps, NSIDE) and weights are 1D with len() = nmaps.

    Parameters
    ----------
    m : healpy map array
        Contains N maps each with the same NSIDE. Shape is (nmaps, NSIDE).
    w : 1D array
        Contains N weights one for each map. Shape is (nmaps, )

    Returns
    -------
    weighted_map : healpy map
        Single healpy map made from the weighted sum of the single maps.
    """
    # We need numpy arrays
    m = np.array(m, ndmin=2)
    w = np.array(w, ndmin=1)
    if len(w) != len(m):
        raise ValueError("Lenghts of map and weight vector don't match.")
    # Maps and weights must match in first shape
    w = w.reshape(len(m), 1)
    # Multiply and sum in axis 0
    return np.sum(m * w, axis=0)


def exp_and_norm_logllh_map(m):
    r"""
    Normalizes a log-likelihhod map m and return the likelihood map.
    Attention: Map must be LOG-LIKELIHOOD, not the negative log-likelihood.

    Some care has to be taken with large or small logllh values because they
    can't be exponentiated to arbitrary precision. See
    `here <http://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability>`_ # noqa
    for more info.

    We use that differences in log translate to ratios in normal space.
    This means our pdf in normal space will be correctly scaled, though we
    subtract stuff in log space.

    1. Subtract max(logllh) from the neglogllh to get values in [*, 0]. This
       has the advantage that very small values get mapped to 0 instead of +inf
       when we exponentiate later.
    2. Then we exponentiate the shifted values and norm them correctly in
       normal space to get a correct normal space pdf.

    Parameters
    ----------
    m : array, healpy map
        Healpy map with LOG-LIKELIHOOD values

    Returns
    -------
    pdf_map : array, healpy map
        Normed normal space Likelihood map, with area under map = 1:

        .. math: \sum_i m_i \cdot \mathrm{d}A = 1
    """
    # 1. Shift values to [*, 0]
    _m = m - np.amax(m)
    # Go from logllh to llh via llh = exp(logllh)
    llhmap = np.exp(_m)
    # Now make pdf from the normal space llh map
    return norm_healpy_map(llhmap)


def norm_healpy_map(m, ret_area=False):
    """
    Norm an arbitrary healpy map to integral over unit sphere to one.
    The discrete integral is then np.sum(pixarea * pixvalue) = 1.

    Returns the normed map and if `ret_area`= True the pixel area for the
    parameters of the given map.
    """
    if hp.maptype(m) != 0:
        raise ValueError(
            "Given map is no healpy map (-1) or a series of maps (>0) : "
            + "{}.".format(hp.maptype(m))
        )

    # Get needed map parameters
    NSIDE = hp.get_nside(m)

    # First normalize to positive values m >= 0 for being a pdf
    if np.any(m < 0):
        m = m - np.amin(m)
    # All pixels are equal area by definition (healpix) so norm to sum
    # of entries and pixel area
    dA = hp.nside2pixarea(NSIDE)
    norm = dA * np.sum(m)
    if norm != 0:
        m = m / norm
    else:
        print("norm is zero. Returning unnormed map.")

    if ret_area:
        return m, dA
    else:
        return m


def get_healpy_map_value(th, phi, m):
    """
    Returns the value at angle (th, phi) from an arbitrary healpy map m.
    Angle th (theta) is something like zenith [0, pi].
    Angle phi is something like azimuth [0, 2pi].
    """
    if hp.maptype(m) != 0:
        raise ValueError(
            "Given map is no healpy map (-1) or a series of maps (>0) : "
            + "{}.".format(hp.maptype(m))
        )

    # Check if th, phi ranges are valid
    if np.any(th) < 0 or np.any(th) > np.pi:
        raise ValueError("th not in range [0, pi].")
    if np.any(phi) < 0 or np.any(phi) > 2 * np.pi:
        raise ValueError("phi not in range [0, 2*pi].")

    # Get needed map parameters
    NSIDE = hp.get_nside(m)
    # Convert all angles to pix indices and get the values at their positions
    pixind = hp.ang2pix(NSIDE, th, phi)
    pdf_vals = m[pixind]

    return pdf_vals


def get_binned_healpy_map_from_idx(idx, NSIDE):
    """
    Get a binned healpy map from given map indices. Uses bincount to bin them
    in healpixels.

    Parameters
    ----------
    idx : array-like
        Integer map indices.
    NSIDE : int
        How many pixels to use for the binning.

    Returns
    -------
    m : array-like, shape(NPIX)
        healpy map with resolution ``NSIDE`` with number of given indices
        falling in the bins specified by ``NSIDE``.
    """
    if not hp.isnsideok(NSIDE):
        raise ValueError("NSIDE has not a valid value of 2^n.")

    # Use bincount to bin the given indices. This is then the final binned map
    NPIX = hp.nside2npix(NSIDE)
    return np.bincount(idx, minlength=NPIX)


def get_binned_healpy_map(theta, phi, NSIDE):
    """
    Get a binned healpy map from single directions.
    Converts directions to indices and uses bincount to bin them in healpixels.

    Parameters
    ----------
    theta : array-like
        healpy ``theta`` angle in ``[0, pi]``.
    phi : array-like, shape (len(theta))
        healpy ``phi`` angle in ``[0, 2pi]``.
    NSIDE : int
        How many pixels to use for the binning.

    Returns
    -------
    m : array-like, shape(NPIX)
        healpy map with resolution ``NSIDE`` with number of given directions
        falling in the bins specified by ``NSIDE``.
    """
    phi = np.atleast_1d(phi)
    theta = np.atleast_1d(theta)
    if not len(theta) == len(phi):
        raise ValueError("`theta` and `phi` must have same length.")
    if not hp.isnsideok(NSIDE):
        raise ValueError("`NSIDE` is not a valid value of form 2^n.")

    # Get pixel indices and use idx binning routine
    idx = hp.ang2pix(theta=theta, phi=phi, nside=NSIDE)
    return get_binned_healpy_map_from_idx(idx, NSIDE)


def gaussian_on_a_sphere(mean_th, mean_phi, sigma, NSIDE, clip=4., log=False):
    r"""
    This function returns a 2D normal pdf on a discretized healpy grid.
    To chose the function values correctly in spherical coordinates, the true
    angular distances to the mean are used.

    Pixels farther away from the mean than clip sigma are clipped because the
    normal distribution falls quickly to zero. The error made by this can be
    easily estimated and a discussion can be found in
    [arXiv:1005.1929](https://arxiv.org/abs/1005.1929v2).

    Parameters
    ----------
    mean_th : float
        Position of the mean in healpy coordinate `theta`. `theta` is in
        [0, pi] going from north to south pole.
    mean_phi : float
        Position of the mean in healpy coordinate `phi`. `phi` is in [0, 2pi]
        and is equivalent to the azimuth angle.
    sigma : float
        Standard deviation of the 2D normal distribution. Only symmetric
        normal pdfs are used here.
    NSIDE : int
        Healpy map resolution :math:`\mathrm{NSIDE} = 2^k, k > 1`.
    clip = float or False or int array, optional
        Wether to clip or take into account all pixels of the map. For high
        resolution the amount of pixels gets very large and the function gets
        slow.

        - If a float ``>0`` is given, the kernel is clipped circular at distance
          ``clip*sigma`` around the mean. If ``<=0`` is given, ``clip``is
          treated as ``False``.
        - If ``clip`` is ``False`` no clipping is applied.
        - If ``clip`` is an array of integer values, only these pixels are
          calculated. Useful, if a custom selection (eg. box) is used.

        (Default: 4.)

    log : bool, optional
        If ``True`` return the log PDF (paraboloid function) instead of the
        normal space gaussian PDF. If log is used, clipped values are the
        minimum of the valid pixels or floating point epsilon. If normal space
        is used, clipped pixel have value 0. (Default: ``False``)

    Returns
    -------
    kernel : array
        Healpy map with resolution NSIDE of the 2D normal distribution. If clip
        was not False, clipped values are set to zero.
    keep_idx : array
        Pixel indices that were actually calculated and not clipped. If clip is
        False this contains all pixel indices for the given resolution NSIDE.
    """
    # Sanity
    if mean_phi > 2. * np.pi or mean_phi < 0.:
        raise ValueError("mean_phi must be in [0, 2pi].")
    if mean_th > np.pi or mean_th < 0.:
        raise ValueError("mean_th must be in [0, pi].")
    mean_phi = np.atleast_1d(mean_phi)
    mean_th = np.atleast_1d(mean_th)

    NPIX = hp.nside2npix(NSIDE)

    if type(clip) is float:
        if clip <= 0:
            # Treat as if False was given and calulate all pixel
            keep_idx = np.arange(NPIX)
        else:
            # Clip circular, keep clip*sigma (radians) around mean
            mean_vec = hp.ang2vec(mean_th, mean_phi)[0]
            # Using inlusive=True to make sure at least on pixel gets returned
            keep_idx = hp.query_disc(
                NSIDE, mean_vec, clip * sigma, inclusive=True)
    elif type(clip) is bool:
        if clip:
            raise ValueError("If `clip` is a bool, it can only be ``False``.")
        keep_idx = np.arange(NPIX)
    else:
        clip = np.atleast_1d(clip)
        if len(clip) > NPIX:
            raise ValueError("Too many `clip` indices for given resolution.")
        if np.any(clip < 0) or np.any(clip > NPIX - 1):
            raise ValueError("`clip` indices outside [0, NPIX-1] range.")
        keep_idx = np.atleast_1d(clip)

    # Create the pixel healpy coordinates
    th, phi = hp.pix2ang(NSIDE, keep_idx)

    # Repeat the fixed vector to calc the distances to
    mean_dir = np.repeat(
        [np.concatenate([mean_th, mean_phi])], repeats=len(keep_idx), axis=0)

    # Get the vectors for all pixels
    all_dir = np.vstack((th, phi)).T

    # For each pixel get the distance to (mean_th, mean_phi) direction
    dist = angdist(mean_dir, all_dir)

    # Get the multivariate normal values at those distances -> kernel func.
    # Gaussian is radial symmetric, so we can use the reduced 2D form
    sigma2 = 2. * sigma**2

    if not log:
        _kernel = np.exp(-dist**2 / sigma2) / np.pi / sigma2
        kernel = np.zeros(NPIX, dtype=float)
    else:
        _kernel = -dist**2 / sigma2 - np.log(np.pi * sigma2)
        # log min values are the minimum > 0 flaot or the smallest kernel val
        min_log = min(np.log(np.finfo(float).eps), np.amin(_kernel))
        kernel = np.ones(NPIX, dtype=float) * min_log

    # Make valid kernel map. If clip, all other pixel are zero / -float.eps
    kernel[keep_idx] = _kernel
    return np.atleast_1d(kernel), np.atleast_1d(keep_idx)


def paraboloid_sigma(llh_map, clip_idx, NSIDE, sigma0, ra0, dec0, loss="chi2"):
    """
    Make a binned LLH fit using a symmetric gaussian PDF correctly transferred
    to the sphere.
    The given pixels of the `llh_map` must stem from a map correctly normalized
    over the unit sphere, so area must be 1. If not possible, interpolate values
    or set them to zero far outside the best fit and use the `norm_healpy_map`
    function to obtain a proper normalization.

    Parameters
    ----------
    llh_map : array-like
        LLH values of the scanned reconstruction algorithm in ra, dec at
        healpy pixel positions indictaed by `clip_idx`. Map must be normalized
        to be a proper PDF on the unit sphere, so integral must be one.
    clip_idx : array-like
        Indices of a whole healpy map at which the LLH was scanned in.
    NSIDE : int
        Healpy resolution of the `lnllh_map` map.
    sigma0 : float
        Initial fit seed for sigma in radian.
    ra0, dec0 : float
        Best fit position of the scanned reconstruction algorithm in
        radian. Determines the fixed positions of the gaussian center.
    loss : str, optional
        Can be one of ``'chi2', 'ce', 'kl'`` for using a chi2, cross-entropy or
        Kullback-Leibler loss function. Result seems to be similar and stable
        for usual cases. (Default: "chi2")

    Returns
    -------
    sigma : float
        Symmetric gaussian sigma which manages to most closely describe the
        scanned, symmetrized LLH.
    """
    # Fixed mean for gaus prior
    th, phi = DecRaToThetaPhi(dec0, ra0)
    if not hp.isnsideok(NSIDE):
        raise ValueError("`NSIDE` is not a valid healpy resolution.")
    if not len(clip_idx) == len(llh_map):
        raise ValueError("Number of map pixels doesn't match given number "
                         + "of clip indices.")
    if hp.nside2npix(NSIDE) < len(llh_map):
        raise ValueError("NSIDE is too small for given map.")

    if loss not in ["chi2", "ce", "kl"]:
        raise ValueError("`loss` can be one of 'chi2', 'ce', 'kl'.")

    if loss == "chi2":
        def loss_fun(sigma):
            """Chi2 loss function using the normal space PDF maps."""
            gaus = gaussian_on_a_sphere(sigma=sigma[0], NSIDE=NSIDE,
                                        mean_phi=phi[0], mean_th=th[0],
                                        clip=clip_idx, log=False)[0][clip_idx]
            return np.sum((llh_map - gaus)**2)
    elif loss == "ce":
        def loss_fun(sigma):
            """Cross entropy loss function."""
            ln_gaus = gaussian_on_a_sphere(sigma=sigma[0], NSIDE=NSIDE,
                                           mean_phi=phi[0], mean_th=th[0],
                                           clip=clip_idx, log=True)[0][clip_idx]
            return -np.sum(llh_map * ln_gaus)
    else:
        # Need the log of the map, so make sure we won't get -inf or nan
        m = (llh_map > 0.)
        llh_map[~m] = np.amin(llh_map[m])
        llh_map = np.log(llh_map)

        def loss_fun(sigma):
            """Kullback-Leibner loss function."""
            ln_gaus = gaussian_on_a_sphere(sigma=sigma[0], NSIDE=NSIDE,
                                           mean_phi=phi[0], mean_th=th[0],
                                           clip=clip_idx, log=True)[0][clip_idx]
            return -np.sum(np.exp(ln_gaus) * (llh_map - ln_gaus))

    res = sco.minimize(loss_fun, x0=[sigma0], jac=False,
                       bounds=[[np.deg2rad(0.01), np.deg2rad(45)]])

    return res


def single_pixel_gaussian_convolution(m, th, phi, sigma, clip=4.):
    r"""
    Calculates the convolution of the pixel on the healpy map `m` with a 2D
    gaussian kernel centered at th, phi with std dev sigma.

    A clipped gaussian kernel function centered at th, phi is generated and
    folded in pixel with the given map for the single pixel at th, phi.
    The convolution is a simple linear combination of map :math:`M` and
    kernel :math:`K` values:

    ..math:

      (\mathrm{Pix}_\mathrm{conv})_i = \sum_j K_j(\theta, \phi)\cdot M_j

    Parameters
    ----------
    m : array
        Valid healpy map.
    th : float
        Position of the mean in healpy coordinate `theta`. `theta` is in
        [0, pi] going from north to south pole.
    phi : float
        Position of the mean in healpy coordinate `phi`. `phi` is in [0, 2pi]
        and is equivalent to the azimuth angle.
    sigma : float
        Standard deviation of the 2D normal distribution. Only symmetric
        normal pdfs are used here. Passed to `gaussian_on_a_sphere()`.
    clip = float
        Wether to clip or take into account all pixels of the map.
        Passed to `gaussian_on_a_sphere()`. (Default: 4.)

    Returns
    -------
    conv_pix : float
        Value of the single pixel of the convolved map at the given position.
    """
    # Sanity
    if phi > 2. * np.pi or phi < 0.:
        raise ValueError("phi must be in [0, 2pi].")
    if th > np.pi or th < 0.:
        raise ValueError("th must be in [0, pi].")

    # Get input map parameters
    NSIDE = hp.get_nside(m)

    # Get the clipped kernel map
    kernel, keep_idx = gaussian_on_a_sphere(
        mean_th=th, mean_phi=phi, sigma=sigma, NSIDE=NSIDE, clip=clip)

    # Normalize kernel after cutoff to ensure normalized pdfs
    kernel = kernel / np.sum(kernel)

    # Now convolve the kernel and the given map for the pixel at pixind only.
    # This is just a linear combination of weighted pixel values selected by
    # the kernel map.
    conv_pix = np.sum(m[keep_idx] * kernel)

    return conv_pix
