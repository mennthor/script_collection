# coding: utf-8

"""
Collection of plotting functions for pyplot functionality in general
"""

import os as _os
import subprocess as subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_working_mpl_backends(which="all", verbose=False):
    """
    Returns a list of working matplotlib backends.

    Note: If you only need a working one, to use mpl functionality but no output
    is needed, eg. the contour functions, use ``matplotlib.use('template')``.

    Parameters
    ----------
    which : str
        Can be one of ``'all', 'interactive', 'non-interactive'``. ``'all'``
        is a concatenation of the other two.
    verbose : bool
        If ``True`` print tested backends during the tests.

    Returns
    -------
    valid_backends : list
        List of valid backend names, which can be used to import ``pyplot``
    all_backends : list
        All matplotlib backends.
    """
    if which == "all":
        backends = mpl.rcsetup.all_backends
    elif which == "interactive":
        backends = mpl.rcsetup.interactive_bk
    elif which == "non_interactive":
        backends = mpl.rcsetup.non_interactive_bk
    else:
        raise ValueError("'which' can be one of 'all', 'interactive' or "
                         + "'non-interactive'")

    valid_backends = []
    cmd = ("python -c 'import matplotlib;matplotlib.use({});"
           + "import matplotlib.pyplot'")
    for b in backends:
        b_str = '"' + b + '"'
        # FNULL, see: https://stackoverflow.com/questions/11269575
        with open(_os.devnull, 'w') as FNULL:
            ret_code = subprocess.call(cmd.format(b_str), shell=True,
                                       stdout=FNULL, stderr=subprocess.STDOUT)
            if ret_code == 0:
                valid_backends.append(b)
            if verbose:
                print("{}: {}".format(b_str, "YES" if ret_code == 0 else "no"))

    return valid_backends, mpl.rcsetup.all_backends


def split_axis(ax, loc="right", size="5%", pad=0.1, cbar=True):
    """
    Splits a matplotlib.Axis using the make_axis_locable function.
    The ticks of the original axis get moved out of the way automatically.

    Parmaters
    ---------
    loc : string
        'right', 'left', 'top', 'bottom'
    size : string
        'd%', where d is the ratio in percent of the original axis size.
    pad : float
        Distance to original axis in inch.
    cbar : bool
        If True (default) strips the right ticks to be used as a cbar.
        Only the `tick_*` parameter must be reset after
        a mappable is given. (colorbar seems to overwrite `tick_*`)

    Returns
    -------
    cax : matplotlib.Axis
        Split off axis.
    """
    div = make_axes_locatable(ax)
    cax = div.append_axes(loc, size=size, pad=pad)

    if loc == "left":
        ax.yaxis.tick_right()
        cax.yaxis.tick_left()
        if cbar:
            cax.xaxis.set_ticks([])
            cax.xaxis.set_ticklabels([])
    elif loc == "right":
        ax.yaxis.tick_left()
        cax.yaxis.tick_right()
        if cbar:
            cax.xaxis.set_ticks([])
            cax.xaxis.set_ticklabels([])
    elif loc == "bottom":
        ax.xaxis.tick_top()
        cax.xaxis.tick_bottom()
        if cbar:
            cax.yaxis.set_ticks([])
            cax.yaxis.set_ticklabels([])
    elif loc == "top":
        ax.xaxis.tick_bottom()
        cax.xaxis.tick_top()
        if cbar:
            cax.yaxis.set_ticks([])
            cax.yaxis.set_ticklabels([])
    else:
        raise ValueError("Unknown position: loc='{}'".format(loc))

    return cax


def hist_from_counts(h, bins, ax=None, **kwargs):
    """
    Makes a hist from existing hist counts.

    Parameters
    ----------
    h : array-like, shape(nbins)
        Counts of the histogram.
    bins : int or array-like, shape (nbins + 1)
        Number of bins or explicit bin edges.
    ax : `matplotlib.axes`, optional
        If not None, plot hist on that axis, otherwise create a new one.
    kwargs
        Directly passed to `matplotlib.pyplot.hist`, without key 'weights'.

    Returns
    -------
    ax : `matplotlib.axes`
        Given or newly created axis on which the hist is plotted.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Remove weights, because we set them ourselves
    kwargs.pop("weights", None)

    mids = 0.5 * (bins[:-1] + bins[1:])
    ax.hist(mids, bins, weights=h, **kwargs)
    return ax
