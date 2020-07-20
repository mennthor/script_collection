# coding: utf-8

"""
Contains rejection sampler for healpy maps.
"""

import numpy as np
import healpy as hp

from .maptools import norm_healpy_map, wrap_theta_phi_range


def healpy_rejection_sampler(m, nsamples, jitter=False, ret_idx=False):
    """
    Sampler to sample positions from given skymaps that are treated as PDFs.

    Pixel indices are drawn according to the pixel weight defined by the map.
    If 'jitter' is True, positions are sampled with a gaussian with width of
    the pixel resolution, to escape from the healpix grid.

    Paramters
    ---------
    m : array-like
        Valid healpy map.
    nsamples : int
        How many sampled to draw from the map.
    jitter : bool, optional
        If True, sample "ungridded" positions as explained above.
        (default: False)
    ret_idx bool, optional
        If True, return the sampled map indices, too. (default: False)

    Returns
    -------
    theta, phi : array-like, shape (nsamples)
        Sampled positions in healpy coordinates.
    idx : array-like, shape (nsamples), optional
        Sampled map indices. If jitter is True, indices may not be the same as
        obtained by ``hp.ang2pix`` using the sampled positions.
        Only returned if `ret_idx` is True.
    """
    if hp.maptype(m) != 0:
        raise ValueError(
            "Given map is no healpy map (-1) or a series of maps (>0) : "
            + "{}.".format(hp.maptype(m)))

    # Make sure the map is in pdf form, which is all positive and area = 1
    m = norm_healpy_map(m)

    # Get weights per pixel from normal space map and normalize
    weights = m / np.sum(m)

    # Draw N pixels from the weighted indices
    NSIDE = hp.get_nside(m)
    NPIX = hp.nside2npix(NSIDE)
    idx = np.random.choice(np.arange(NPIX), size=nsamples, p=weights)

    # Get angles from pix indices
    theta, phi = hp.pix2ang(NSIDE, idx)

    # Sample new angles from a gaussian with width of half the pixel resolution.
    # Otherwise we get angles on a grid, exactly one position from one pixel.
    if jitter:
        res = hp.nside2resol(NSIDE) / 2.
        theta = np.random.normal(theta, res)
        phi = np.random.normal(phi, res)
        theta, phi = wrap_theta_phi_range(theta, phi)

    if ret_idx:
        return theta, phi, idx
    else:
        return theta, phi


def healpy_rejection_sampler_OLD(m, n):
    """
    Special version of the generic rejection sampler. Here we sample map
    indices uniformly across the map (range [0, pi]x[0, 2*pi]) and check if
    the rejection criterium is fullfilled at that point.
    We can sample indices uniformly because healpixel areas are are equal
    across the sphere.

    Rejection algorithm: For each point draw a uniform rand number and reject
    the point if the value of the pdf is smaller. Else accept it and repeat
    until the n points are sampled.

    Returns n map indices drawn from the map transformed to a pdf.
    The resolution is the same as the input map which is sampled from.
    """
    if hp.maptype(m) != 0:
        raise ValueError(
            "Given map is no healpy map (-1) or a series of maps (>0) : "
            + "{}.".format(hp.maptype(m))
        )

    # Get map parameter
    NPIX = hp.get_map_size(m)
    # Make sure the map is in pdf form, which is all positive and area = 1
    m = norm_healpy_map(m)
    # Get the pdf maximium to set the sampling bounding box
    fmax = np.amax(m)

    def pdf(ind):
        """Simple wrapper to consistently use the name pdf.
        Returns the mapvals at given indices"""
        return m[ind]

    # Create list to store the sampled events
    sample = []
    # Rejection sampling loop
    nstart = n
    efficiency = 0
    while n > 0:
        # Count trials for efficinecy, has nothing to do with the sampling
        efficiency += n

        # Only choose integer map indices randomly as we operate on discrete
        # maps
        r1 = np.random.randint(0, NPIX, size=n)
        # Create n uniform distributed rand numbers between 0 and the maximum
        # of the function
        r2 = fmax * np.random.uniform(0, 1, n)
        # Calculate the pdf value pdf(r1) and compare to r2
        # --> If r2 is below or equal func(r1) accept the event, else reject it
        accepted = (r2 <= pdf(r1))
        # --> Where func(r2) is near the box boundary of fmax the efficiency is
        #     good, else it gets worse. Append only accepted random numbers
        sample += r1[accepted].tolist()
        # Redo with n = events that are missing until n = 0 and all n requested
        # events are generated
        n = np.sum(~accepted)
    # eff first counts how many events where drawn from the pdf and is then the
    # fraction of the actual desired events n by the generetaded events.
    efficiency = nstart / float(efficiency)
    return np.array(sample), efficiency
