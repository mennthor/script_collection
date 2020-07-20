# coding: utf-8

"""
Script to test the anapymods.healpy.add_weighted_maps() function by creating two
gaussian maps and add them with different weights.
"""

import numpy as np
import matplotlib.pyplot as plt

from script_collection.healpy.maptools import (
    gaussian_on_a_sphere, add_weighted_maps)
from script_collection.plots.astro import mollview

NSIDE = 64

# Create maps
kern1 = gaussian_on_a_sphere(
    np.pi / 2., 3 * np.pi / 2., np.deg2rad(20), NSIDE, clip=4.0)
kern2 = gaussian_on_a_sphere(
    np.pi / 2., np.pi / 2., np.deg2rad(20), NSIDE, clip=4.0)

map1 = kern1[0]
mollview(map1, cmap="Greens")
plt.title("Map1 at ra = 3/2*pi")

map2 = kern2[0]
mollview(map2, cmap="Greens")
plt.title("Map2 at ra = pi/2")

# Add them creating some dipol like structure
weighted_map = add_weighted_maps(
    np.array([map1, map2]), w=np.array([-1, 1]))
mollview(weighted_map, cmap="coolwarm")
plt.title("Added with array. Weights: M1=-1, M2=1")

plt.show()
