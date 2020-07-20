# coding:utf8

"""
Test the mollview_grid() plotter from anapymods/plotting/_astro.py.
The function takes a 2D array of healpy maps and plots them all at once on
a grid defined by the mapp array.
"""

import numpy as np
import matplotlib.pyplot as plt

from script_collection.healpy.maptools import gaussian_on_a_sphere
from script_collection.plots.astro import mollview_grid

# Make two maps. One with a bg and a point source and on ewith 2 point srcs
log_hor_ = np.arange(12 * 32**2, dtype=np.float) / 1000.
log_hor_ += gaussian_on_a_sphere(mean_phi=0.7 * np.pi,
                                 mean_th=0.8 * np.pi / 2.,
                                 sigma=np.deg2rad(5),
                                 NSIDE=32, clip=False)[0]
log_equ_, _ = gaussian_on_a_sphere(mean_phi=1.1 * np.pi,
                                   mean_th=1.1 * np.pi / 2.,
                                   sigma=np.deg2rad(5),
                                   NSIDE=32, clip=False)
log_equ_ += gaussian_on_a_sphere(mean_phi=0.7 * np.pi,
                                 mean_th=0.8 * np.pi / 2.,
                                 sigma=np.deg2rad(5),
                                 NSIDE=32, clip=False)[0]

# Arrange in 2 columns and 3 rows
_maps = np.array([[log_hor_, log_equ_],
                  [log_hor_, log_equ_],
                  [log_hor_, log_equ_]])

_labels = np.array([["ul", "ur"],
                    ["ml", "mr"],
                    ["bl", "br"]])

_proj = np.array([["mollweide", "hammer"],
                  ["hammer", "mollweide"],
                  ["mollweide", "hammer"]])

# Create all plots
fig, ax = mollview_grid(
    _maps, labels=_labels, projections=_proj, renderpix=300)

# Test correct location of axes in array
ax[1, 1].set_title("Test Mid Right Title")
ax[2, 0].set_title("Test Bottom Left Title")

fig.set_facecolor("w")
fig.tight_layout()
plt.show()
