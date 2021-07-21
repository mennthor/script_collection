# coding: utf-8

"""
Collection of plotting functions for pyplot functionality in general
"""

import os as _os
import subprocess as subprocess

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def hist_outline(x, ax, outl_kwargs={}, hist_kwargs={}):
    """
    Creates a 1D histogram exactly like `matplotlib.pyplot.hist` but with an
    additional outline along the histogram bars.

    Parameters
    ----------
    x : array or sequence of arrays
        Input values exactly as in `matplotlib.pyplot.hist`.
    ax : `matplotlib.axes.Axes`
        The axes to add the histogramm to.
    outl_kwargs : dict of keyword args
        Arguments passed to `matplotlib.pyplot.plot` to control the outline
        looks. The key 'drawstyle' is always overwritten with 'steps-pre'.
        If color and alpha arguments are not given, the ones from the histogram
        are used (if given there) and the alpha is increased by a factor of 1.5.
    hist_kwargs : dict of keyword args
        Arguments passed to `matplotlib.pyplot.hist` controlling the histogram
        looks.

    Returns
    -------
    h, b, pc
        The unaltered return values of `matplotlib.pyplot.hist` (histogram
        array, bin array and patch collection for the drawn bars).
    lc
        The unaltered return value of `matplotlib.pyplot.plot` (line collection
        for the drawn line).

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.random.exponential(size=1000)
    >>> fig, ax = plt.subplots(1, 1)
    >>> out = hist_outline(
            x, ax, outl_kwargs={"c": "k", "lw": 2},
            hist_kwargs={"color": "C7","bins": 50, "density": True})
    >>> ax.set_yscale("log")
    >>> plt.show()
    """
    # Make the histogram
    h, b, pc = ax.hist(x, **hist_kwargs)

    # Use hist color and alpha of not explicetly given in 'outl_kwargs'
    if not any(k in outl_kwargs for k in ("c", "color")):
        try:
            outl_kwargs["color"] = hist_kwargs["color"]
        except KeyError:
            pass  # Not present in hist_kwargs either
    if "alpha" not in outl_kwargs:
        try:
            outl_kwargs["alpha"] = hist_kwargs["alpha"] * 1.5
        except KeyError:
            pass  # Not present in hist_kwargs either

    # Draw the outline
    outl_kwargs["drawstyle"] = "steps-pre"
    lc = ax.plot(np.r_[b[0], b, b[-1]], np.r_[0, h[0], h, 0], **outl_kwargs)

    return h, b, pc, lc


def add_ellipsis(ax, x0, y0, r0, r1, logx=False, **plt_kwargs):
    """
    Add ellipsis to axis. Can handle lin-log plots (x-axis is log10).
    Cannot handle rotations or y axis is on log scale.

    Parameters
    ----------
    ax : axis
        The matplotlib axis to draw on.
    x0, y0 : float
        The center coordinates of the ellipsis. If `logx` is `True`, `x0` must
        be `>0`.
    r0, r1 : float
        The x and y radii of the ellipsis.
        If `logx` is `True`, then this in log coordinates (so an exponential
        width), else it is in normal linear coordinates.
    logx : bool, optional (default: False)
        Wether to draw the ellipse in log x-axis or lin x-axis

    Returns
    -------
    l_lo, l_up : line collection
        The drawn lines of the lower and upper ellipsis half.

    Example
    -------
    ```
    import matplotlib.pyplot as plt
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8,3))
    add_ell(axl, 1, 0, 0.75, 1, logx=True, c="k", label="Log ell")
    add_ell(axl, 1, 0, 0.75, 1, logx=False, c="C3", label="Lin ell")
    axl.set_xscale("log")
    axl.set_xlim(1e-1, 1e1)
    axl.set_ylim(-1.25, 1.25)
    axl.set_title("Lin x scale")
    add_ell(axr, 1, 0, 0.75, 1, logx=True, c="k", label="Log ell")
    add_ell(axr, 1, 0, 0.75, 1, logx=False, c="C3", label="Lin ell")
    axr.set_xlim(0, 2)
    axr.set_ylim(-1.25, 1.25)
    axr.set_title("Log x scale")
    fig.tight_layout()
    plt.show()
    ```
    """
    # When x is log, just treat dx as it would be in exponent linear space
    dx = np.linspace(-r0, r0, 250)
    y_lo = y0 - r1 / r0 * np.sqrt(r0**2 - dx**2)
    y_up = y0 + r1 / r0 * np.sqrt(r0**2 - dx**2)
    if logx:
        # But now convert x coords back to linear space
        l_lo = ax.plot(10**(np.log10(x0) + dx), y_lo, **plt_kwargs)
        l_up = ax.plot(10**(np.log10(x0) + dx), y_up, **plt_kwargs)
    else:
        # For linspace, everything normal
        l_lo = ax.plot(x0 + dx, y_lo, **plt_kwargs)
        l_up = ax.plot(x0 + dx, y_up, **plt_kwargs)
    return l_lo, l_up


def set_log10_ticks(ax, lo, hi, minor=False, fmter=(None, None)):
    """
    Sets nicely (NOT evenly) spaced ticks for the given axis in log10 scale.

    Parameters
    ----------
    ax : matplotlib.axis.Axis
        x or y axis to set the ticks for (eg. `plt.gca().xaxis`).
    lo, hi : float
        Upper and lower tick boundary.
    minor : bool, optional (default: False)
        If `True` also locate minor ticks.
    fmter : tuple, optional (default: (None, None))
        Custom formatters for major, minor ticks. Minor formatter is only used
        if `minor` is `True`.
    """
    ndecs = np.ceil((np.log10(hi))) - np.floor((np.log10(lo)))
    if ndecs < 1:
        raise ValueError("[lo, hi] range needs to span a decade.")
    fmt_maj = mticker.ScalarFormatter() if fmter[0] is None else fmter[0]
    fmt_min = mticker.NullFormatter() if fmter[1] is None else fmter[1]

    t_maj = mticker.LogLocator(  # E.g. [10, 20, 30, ..., 90, 90, 100]
        base=10, subs=np.arange(0.1, 1., 0.1), numticks=ndecs + 1)
    ax.set_major_locator(t_maj)
    ax.set_major_formatter(fmt_maj)
    t_min = mticker.LogLocator(  # E.g. [10, 11, 12, ..., 98, 99, 100]
        base=10, subs=np.arange(0.01, 1., 0.01), numticks=ndecs + 1)
    if minor:
        ax.set_minor_locator(t_min)
        ax.set_minor_formatter(fmt_min)


def make_log_ticks(log_lo, log_hi, cut_lo=None, cut_hi=None):
    """
    Makes linear equally spaced major and minot tick steps in each decade.
    Eg. between 10 and 100 makes major steps `[10, 20, ..., 90, 100]` and minors
    `[10, 11, 12, ..., 98, 99, 100]`.

    Note: The ticks need to be set using set_[x,y]ticks, but it seems,
          matplotlib does not like this when using the minor ones...
          THere is a working version using the `ticker` functions though.
          See `set_log10_ticks` above.

    Parameters
    ----------
    log_lo, log_hi : int
        Lower, upper bounds in log10 scale (inclusive).
    cut_lo, cut_hi : float
        Actual lower, upper bounds in linear scale.

    Returns
    -------
    major, minor : list
        Major and minor tick lists. Minor ticks are space by `decade // 100`,
        major ones by `decade // 10`.
    """
    if cut_lo is None:
        cut_lo = 10**log_lo
    if cut_hi is None:
        cut_hi = 10**log_hi

    ticks, minor = [cut_lo], [cut_lo]
    ndecs = 10**np.arange(log_lo, log_hi + 1, 1)
    for dec in ndecs:
        if ticks[-1] > dec:
            continue

        step, step_min = dec // 10, dec // 100
        ticks += list(np.arange(
            ticks[-1] + step, min(dec, cut_hi) + step, step))
        minor += list(np.arange(
            minor[-1] + step_min, min(dec, cut_hi) + step_min, step_min))

    return ticks, minor
