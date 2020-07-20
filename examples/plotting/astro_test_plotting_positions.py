# coding:utf8

"""
Test the combination of healpy map plotting and normal plot function.

The coordinate transformation functions should convert properly from
healpy to map coordinates.

We set one pixel to be the maximum and plot a marker at this position.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from script_collection.plots.astro import mollview, ThetaPhiToMapCoords

# Create a simple map
NSIDE = 8
NPIX = 12 * NSIDE**2
m = np.ones(NPIX)

# Set one pixel as maximum
m[200] = 2
m_max_pix = np.argmax(m)

# Plot the map
mollview(m, cmap="viridis")

# Convert the healpy coordinates from the maximum to map cooridnates
th, phi = hp.pix2ang(NSIDE, m_max_pix)
x, y = ThetaPhiToMapCoords(th, phi)

# Use normal plot function
plt.plot(x, y, "w*", ms=8, mec="k", mew=1)

plt.title("phi, th : {}, {} deg | y, x: {}, {} deg".format(
    np.rad2deg(phi), np.rad2deg(th), np.rad2deg(x), np.rad2deg(y)))

plt.show()
