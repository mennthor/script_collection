# coding:utf8

"""
Testing the quick view functions `mollview()`and `cartview()` that provide
similar functionality than the equally named healpy functions.
"""

import numpy as np
import matplotlib.pyplot as plt

from script_collection.plots.astro import mollview, cartview


# Make simple healpy map. NPIX = 12 * NSIDE**2
m = np.arange(12 * 2**2)

# View in mollweide projection skymap
fig, ax = mollview(m)
fig.set_facecolor("w")

# View in rectilinear projection
fig, ax = cartview(m)
fig.set_facecolor("w")

plt.show()
