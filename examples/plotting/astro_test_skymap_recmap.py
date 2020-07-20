# coding:utf8

"""
Test script for the anapymods plotting astro functions.

We make two subplots and plot a healpy map on each.
First using a rectinlinear grid and then using a skymap mollweide projection.

Turn TeX True or False to see the difference in the rendering options for
the skymap figure.
"""

import numpy as np
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from script_collection.plots.astro import recmap, skymap, plot_healpy_map
from script_collection.plots.colors import dg


TeX = True
if TeX:
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = "cm"

# Get the plotting classes
recmap = recmap()
skymap = skymap()

# Make gridspec with two plots vertically stacked
grid = gridspec.GridSpec(nrows=2, ncols=1)
fig = plt.figure(figsize=(8, 6), facecolor="w")

# Create the healpy map to plot
NSIDE = 8
NPIX = hp.nside2npix(NSIDE)
m = np.arange(NPIX)

# Rectilinear plot on top
fig, axt = recmap.figure(
    fig=fig,
    gs=grid[0],
    gal_plane=True,
    xticks=np.arange(-180, 180 + 30, 30),
    yticks=np.arange(-80, 80 + 20, 20),
)

# Let the function determin how many pixel to render
fig, axt = plot_healpy_map(m, ax=axt, cbar_orient="vertical",
                           cbar_label="map value", renderpix=0)

axt.set_xlabel("right-ascension in hours")
axt.set_ylabel("declination in degree")

# This is a bug: After plotting the colormesh the grid disappears, but only
# in the rectilinear plot. The skymap works as expected.
axt.grid(True, ls=":", color=dg)

# Skymap plot at the bottom
fig, axb = skymap.figure(
    fig=fig,
    gs=grid[1],
    proj="mollweide",
    gal_plane=True,
    tex=mpl.rcParams["text.usetex"],
    xticks=np.arange(-180, 180 + 30, 30),
    yticks=np.arange(-80, 80 + 20, 20),
)

# Render 200 pixel on pcolormesh
fig, axb = plot_healpy_map(m, ax=axb, renderpix=200, cbar_orient="horizontal")

fig.tight_layout()
plt.show()
