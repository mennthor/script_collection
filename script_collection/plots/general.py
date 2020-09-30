# coding: utf-8

"""
Collection of plotting functions for pyplot functionality in general
"""

import os as _os
import subprocess as subprocess

import numpy as np
import scipy.stats as scs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
import colorsys
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


def add_poisson_residual_plot(
    counts, mus, alphas=[0.68, 0.95, 0.99], bins=None,
        colors="YlOrRd_r", ax=None, ylabel_kw={}, plot_kw={}):
    """
    Adds plot to given axis that shows the data counts location in the central
    poisson interval region for the corrsponding expectation in the same bin and
    the given probability.

    Parameters
    ----------
    counts : array-like
        Counts per bin. Counts must be >= 0.
    mus : array-like
        Poisson expectation values (sometimes also 'lambda') for each bin.
        Must be >= 0. If zero, then the corresponding count must also be zero.
    alphas : array-like, optional
        Probability content (`0 < alpha < 1`) in the central interval that are
        plotted. The intervals are sorted, made unique and plotted centered
        around the CDF value of the expectation in each bin and are scaled by
        this, so that when a count equal the expectation, it is plotted at 1
        (kind of to stick to the `(count - mean) / stddev` plots.).
        (default: [0.68, 0.95, 0.99])
    colors : str or list, optional
        If a single string is given, it is interpreted as a matplotlib colormap
        name and a color value per alpha is sampled from that colormap. If the
        sampled color for the outermost intervall would be white, the colormap
        is started at 0.1 instead of zero.
        If a list, each entry is expected to be in a valid matplotlib color
        format and the number of colors must match the number of alphas.
        (default: 'YlOrRd')
    ax : matplotlib.axes.Axes
        If `None`, a new figure with a single axis is returned. Otherwise the
        plot is added to the given axis. (default: None)

    Returns
    -------
    intervals : list
        List of `[lo_array, hi_array]` per given, unique interval, where the
        arrays contain the lower and upper bound of the central alpha interval
        for each given expectation value.
    ax : matplotlib.axes.Axes
        The axis the plot was drawn to.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> counts = [0, 0, 3, 3, 2, 0, 6, 7, 9, 10, 0, 13, 18, 24, 30]
    >>> mus = [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    >>> intervals, ax = add_poisson_residual_plot(
    >>>     counts=counts, mus=mus, alphas=[0.68, 0.95, 0.99])
    >>> plt.show()
    """
    mus = np.atleast_1d(mus)
    counts = np.atleast_1d(counts).astype(int)
    if np.any(counts < 0):
        raise ValueError("Negative counts given.")

    if len(counts) != len(mus):
        raise ValueError("`counts` and `mus` arrays must have same length.")

    if bins is None:
        bins = np.arange(len(counts) + 1)  # If not given, simply use indices
    else:
        bins = np.atleast_1d(bins)
        if len(counts) != len(bins) - 1:
            raise ValueError("`bins` array must have length `len(counts) + 1`.")

    # Make new figure if no axis was given
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # Manage colors: If str assume cmap, if list, assume explicit colors
    if isinstance(colors, str):
        # Check if cmap starts with white, if yes, clip to be visible against BG
        cmap = plt.get_cmap(colors)
        _start = 0
        if np.allclose(cmap(0.), 1.):
            _start = 0.1
        colors = cmap(np.linspace(_start, 1., len(alphas)))
    else:
        colors = np.atleast_1d(colors)
        if len(colors) != len(alphas):
            raise ValueError("Explicit list of colors given, but its length "
                             "does not match number of intervals.")

    # Get the alpha values for the expectations. Note: Alphas for mu go to 0.5
    # only when the expectation tends towards inf. Usually they are different
    INVALID_VAL_LO = 1e-300
    alpha_mus = scs.poisson.cdf(mus, mus)
    # These have no drawable interval, set valid, but don't draw them later
    mask_alpha_mu_zero = np.isclose(mus, 0.)
    alpha_mus[mask_alpha_mu_zero] = 0.5
    # There must be no counts where no expectation is
    if np.any(counts[mask_alpha_mu_zero] > 0.):
        raise ValueError("There can't be counts when the expectation is zero.")

    # Create intervals
    alphas = np.unique(alphas)
    if not np.all(np.logical_and(alphas > 0, alphas < 1)):
        raise ValueError("Interval alphas must be in (0, 1).")
    intervals = []
    for alpha in alphas:
        intervals.append(scs.poisson.interval(alpha, mus))

    # Plot confidence bands with largest alpha first to avoid occlusion
    # TODO: Handle empty mus (just show data point in there, which shoold also
    # better be zero, because a poisson with mu = 0 can't have counts)
    # In a first pass, compute and prepare all values for custom log plot
    max_log_range = 0.
    for i, (low, hig) in enumerate(intervals[::-1]):
        # Because the Poisson distribution is discrete, we have to remove the
        # lower border of the intervall manually, otherwise too much probability
        # is included (the intervals are inclusive in both directions, but the
        # CDF is right inclusive, so for low we need to remove the endpoint).
        alpha_lo = scs.poisson.cdf(low, mus) - scs.poisson.pmf(low, mus)
        alpha_hi = scs.poisson.cdf(hig, mus)

        # To have a consistent y-scale and to center the expectation value
        # (which is only at alpha=0.5 for large mus) in the center, we scale by
        # the alpha of the expectation. Then we use the double, half reverse log
        # scale, because otherwise we can"t see the higher p-value intervals.
        alpha_lo = alpha_lo / alpha_mus  # [0, a_mu]->[0, 1]
        # [a_mu, 1]->[0, 1]
        alpha_hi = (alpha_hi - alpha_mus) / (1. - alpha_mus)
        alpha_lo[np.isclose(alpha_lo, 0.)] = 0.  # Some numerics at lower edge
        assert not np.any(alpha_lo[~mask_alpha_mu_zero] < 0)
        assert not np.any(alpha_lo[~mask_alpha_mu_zero] > 1)
        assert not np.any(alpha_hi[~mask_alpha_mu_zero] < 0)
        assert not np.any(alpha_hi[~mask_alpha_mu_zero] > 1)

        # Clip invalid logs to small float (only happens when lo is at 0.)
        inv_log_mask = np.isclose(alpha_lo, 0.)
        alpha_lo[inv_log_mask] = INVALID_VAL_LO

        # Now put them on a log scale for low and on a "reverse" log scale for
        # high to zoom to the regions where the high p-value intervals are
        # Normal log, zoom towards 0, these are in normal log range [R-, 0]
        log_alpha_lo = np.zeros_like(alpha_lo)
        log_alpha_hi_inv = np.zeros_like(alpha_hi)
        log_alpha_lo[~mask_alpha_mu_zero] = np.log10(
            alpha_lo[~mask_alpha_mu_zero])
        # Reverse log, zoom towards 1, invert to range [0, R+]
        log_alpha_hi_inv[~mask_alpha_mu_zero] = -1. * np.log10(
            1. - alpha_hi[~mask_alpha_mu_zero])

        # Store max valid range for plot ylims
        max_log_range = max(
            max_log_range,
            max(np.amax(np.abs(log_alpha_lo[~inv_log_mask])),
                np.amax(log_alpha_hi_inv)))

        # Plot intervalls:
        # Fill between interval borders in scaled axis. We fill the low values
        # from inf to 0 and the reversed high ones from 0 to inf
        _la_lo = np.append(log_alpha_lo[0], log_alpha_lo)
        _la_hi = np.append(log_alpha_hi_inv[0], log_alpha_hi_inv)
        _zeros = np.zeros_like(_la_lo)
        ax.fill_between(bins, _la_lo, _zeros, step="pre", color=colors[i])
        ax.fill_between(bins, _zeros, _la_hi, step="pre", color=colors[i])

    # Set center line w or k depending on inner color lightness
    _, lightness, _ = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(colors[-1]))
    _c = "k"
    if lightness < 0.25:
        _c = "w"
    ax.axhline(0, 1, 0, c=_c, lw=2, ls="-")

    # Plot data points as dots w/o errors and in the alpha units from above.
    # Override defaults if given (remove some non-unique settings first)
    markers = {"marker": "o", "marker_lo": "v", "marker_hi": "^",
               "marker_zero": "o"}  # Keep separate
    plot_kwargs = {
        "color": "w",
        "ls": "",
        "ms": "10",
        "markeredgewidth": 1.5,
        "markeredgecolor": "k",
    }
    plot_kwargs.update(plot_kw)
    # Replace shortcuts to avoid double settings (there is a mpl decorator for
    # this, but in which module is it?)
    _repl = {
        "mew": "markeredgewidth", "mec": "markeredgecolor",
        "mfc": "markerfacecolor", "m": "marker", "linestyle": "ls",
        "markersize": "ms", "c": "color"}
    for k, v in _repl.items():
        if k in plot_kwargs:
            plot_kwargs[v] = plot_kwargs[k]
            del plot_kwargs[k]
    _del = []
    for name, val in plot_kwargs.items():  # Update marker dict separately
        if name in markers:
            markers[name] = val
            _del.append(name)
    for name in _del:
        del plot_kwargs[name]
    # No data point is plotted (is 0 anyway) where expectation is zero
    alpha_mus = alpha_mus[~mask_alpha_mu_zero]
    alpha_cnts = scs.poisson.cdf(counts, mus)[~mask_alpha_mu_zero]
    # Pretty sure we should include these, the CDF is not always  0 for 0 cnts
    # alpha_cnts[counts[~mask_alpha_mu_zero] == 0] = INVALID_VAL_LO
    mids = 0.5 * (bins[1:] + bins[:-1])[~mask_alpha_mu_zero]
    # Case 1/2: Counts alpha >= alpha_mu
    m_hi = (alpha_cnts >= alpha_mus)
    # Transform to inverted log vals
    alpha_cnts_hi = alpha_cnts[m_hi]
    alpha_cnts_hi = (alpha_cnts_hi - alpha_mus[m_hi]) / (1. - alpha_mus[m_hi])
    log_alpha_cnts_hi_inv = -1. * np.log10(1. - alpha_cnts_hi)
    # Outside range? Plot with 'marker_hi' at upper bound, else use 'marker'
    _m = (log_alpha_cnts_hi_inv > max_log_range)
    ax.plot(mids[m_hi][~_m], log_alpha_cnts_hi_inv[~_m],
            marker=markers["marker"], **plot_kwargs)
    ax.plot(mids[m_hi][_m], np.sum(_m) * [max_log_range],
            marker=markers["marker_hi"], **plot_kwargs)
    # Case 2/2: Counts alpha < alpha_mu
    m_lo = (alpha_cnts < alpha_mus)
    # Transform to log vals
    alpha_cnts_lo = alpha_cnts[m_lo]
    alpha_cnts_lo = alpha_cnts_lo / alpha_mus[m_lo]
    log_alpha_cnts_lo_inv = np.log10(alpha_cnts_lo)
    # Outside range? Plot with 'marker_lo' at lower bound
    _m = (log_alpha_cnts_lo_inv < -max_log_range)
    ax.plot(mids[m_lo][_m], np.sum(_m) * [-max_log_range],
            marker=markers["marker_lo"], **plot_kwargs)
    # Zero counts but in range? Use 'marker_zero' at alpha, else use 'marker'
    _m0 = np.logical_and(~_m, counts[~mask_alpha_mu_zero][m_lo] == 0)
    ax.plot(mids[m_lo][_m0], log_alpha_cnts_lo_inv[_m0],
            marker=markers["marker_zero"], **plot_kwargs)
    _mrest = np.logical_and(~_m, counts[~mask_alpha_mu_zero][m_lo] > 0)
    ax.plot(mids[m_lo][_mrest], log_alpha_cnts_lo_inv[_mrest],
            marker=markers["marker"], **plot_kwargs)

    # Plot setting
    # Build custom ticks in logspace
    def ticker(x, pos):
        """ Make log(abs(x)) ticks. x is tick value, pos its position """
        if x == 0:
            return "$1$"
        return "$10^{{-{:.0f}}}$".format(np.abs(x))

    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ticker))
    # This sets the ticklabels, so we can get their widths below
    ax.get_figure().canvas.draw()

    # Build split ylabel
    # Set empty and custom text as proper label
    ax.set_ylabel("")
    # Find lowest x bbox for y ticklabels to place the custom text there
    # From: https://stackoverflow.com/questions/5320205/matplotlib-text-dimensions
    renderer = ax.get_figure().canvas.get_renderer()
    _xmin_ax = 0
    for text in ax.get_yticklabels():
        bb = text.get_window_extent(renderer=renderer)
        bbt = matplotlib.transforms.Bbox(ax.transAxes.inverted().transform(bb))
        _xmin_ax = min(_xmin_ax, bbt.bounds[0])
    # Override defaults if given
    ylabel_kwargs = {
        "x": _xmin_ax,
        "y": 0.5,
        "s": (r"$\alpha / \alpha_{{\mu}} \enspace\leftrightarrow\enspace "
              r"p / \alpha_{{\mu}}$"),
        "rotation": 90,
        "ha": "center",
        "va": "center",
        "transform": ax.transAxes,
    }
    ylabel_kwargs.update(ylabel_kw)
    ax.text(**ylabel_kwargs)
    ax.set_xlim(bins[0], bins[-1])
    max_log_range = (np.round(max_log_range, decimals=1) + 0.1) * 1.1
    ax.set_ylim(-max_log_range, max_log_range)

    return intervals, ax
