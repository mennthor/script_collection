"""
Making residual plots with arbitrary underlying PDFs.
See: https://github.com/tudo-astroparticlephysics/pydisteval for a more
professional approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
import colorsys


def add_residual_plot(
    dist, counts, alphas=(0.68, 0.95, 0.99), dist_kwargs={}, atol=1e-11,
        bins=None, colors="YlOrRd_r", ax=None, plot_kw={}):
    """

    Parameters
    ----------
    dist : callable
        Distribution to use for the intervals. Is called as
        `a_counts, a_expect, intervalls = dist(counts, alphas, **dist_kwargs)`
        and must return alpha values for the counts, for the expectation value
        and upper, lower interval borders for each alpha and each count. The
        intervalls also have to be represented in CDF values (alphas).
    counts : array-like
        Number of data counts per bin.
    alphas : tuple, optional
        Tuple of probability contents of the central intervals to plot.
        (default: (0.68, 0.95, 0.99))
    dist_kwargs : dict, optional
        Additional arguments passed to the `dist` callable. (default: {})
    atol : float, optional
        Value under which interval boundaries are clipped. This is used to clip
        intervals that are non-zero due to numerical errors which avoids
        squashing all the proper (and smaller intervals) to a narrow center
        region in the plot. (default: 1e-11)
    bins : array-like, optional
        Used to mark the xticks in the plot. If `None`, `range(0, len(counts))`
        is used. (default: `None`)
    colors : str or list, optional
        The colors to use for each interval. If a list, there must be a valid
        `matplotlib` color for each given interval. If a single string, it is
        interpreted as a `matplotlib` colormap from which interval colors are
        sampled. (default: "YlOrRd_r")
    ax : matplotlib.axes.Axes
        The axis to add the plot to. If `None` a new standard figure is created.
        (default: `None`)
    plot_kw : dict
        Additional arguments passed to the plotting function for the data
        points (`matplotlib.pyplot.plot`). Additionally to the standard keys,
        the keys 'marker_lo', 'marker_hi', 'marker_zero' can be set. The accept
        a `matplotlib.marker` srtring and chose the marker for data points being
        below the y-limits, above the y-limits and points with zero counts.
        Default markers are: `{"marker": "o", "marker_lo": "v",
        "marker_hi": "^", "marker_zero": "o"}`.  (default: {})

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis on which the plot was drawn.
    alpha_cnts : array-like
        CDF values for each counts from the given distribution.
    alpha_expect : array-like
        CDF values for the expectation values per bin from the given
        distribution.
    alpha_intervals : array-like
        CDF values for each `(lo, hi)` edges for all bins per given intervall
        from the given distribution.

    Example
    -------
    >>> from script_collection.plots.residuals import add_residual_plot
    >>> from script_collection.stats.distributions import dist_poisson
    >>> import matplotlib.pyplot as plt
    >>> counts = np.array([0, 0, 3, 3, 2, 0, 6, 7, 9, 10, 0, 13, 18, 24, 130])
    >>> mus = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 12, 13, 14, 100])
    >>> ax, alpha_cnts, alpha_expect, alpha_intervals = add_residual_plot(
    >>>     counts=counts, dist=dist_poisson, alphas=[0.68, 0.95, 0.99],
    >>>     dist_kwargs=dict(mus=mus))
    >>> plt.show()
    """
    CLIP_LOG = 1000  # Clipping log(0) so they still show, but are not inf

    # Check and prepare input data
    counts = np.atleast_1d(counts).astype(int)
    if np.any(counts < 0):
        raise ValueError("Negative counts given.")

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

    # #########################################################################
    # Get CDF values for counts, expectations (for scaling) and intervals
    alpha_cnts, alpha_expect, alpha_intervals = dist(
        counts, alphas, **dist_kwargs)
    if np.any(np.logical_or(alpha_cnts < 0., alpha_cnts > 1.)):
        raise ValueError(
            "`dist` returned CDF values outside [0, 1] for the given counts.")
    if np.any(np.logical_or(alpha_expect < 0., alpha_expect > 1.)):
        raise ValueError(
            "`dist` returned CDF values outside [0, 1] for the expectations.")
    for alpha, (los, his) in zip(alphas, alpha_intervals):
        if np.any(los > his):
            raise ValueError("`dist` returned flipped CDF values for the "
                             "{:.2g} interval.".format(alpha))
        if np.any(np.logical_or(los < 0., his > 1.)):
            raise ValueError("`dist` returned CDF values outside [0, 1] for "
                             "the {:.2g} interval.".format(alpha))

    # #########################################################################
    # Plot intervals with largest alpha first to avoid occlusion. The intervals
    # are plotted in a double log scale, so that the region around alpha = 0 and
    # alpha = 1 are zoomed into
    max_log_range = 0.  # For setting plot limits later
    valid_intervals = np.zeros(len(counts), dtype=bool)
    for i, (alpha_lo, alpha_hi) in enumerate(alpha_intervals[::-1]):
        # Ignore log(0) (0 divide) and 0/0 (nan), they are filtered afterwards
        with np.errstate(divide="ignore", invalid="ignore"):
            # To have a consistent y-scale and to center the expectation value,
            # we scale by the alpha of the expectation. Clip to avoid numerical
            # errors. Lo: [0, a_mu]->[0, 1], Hi: [a_mu, 1]->[0, 1]
            alpha_lo = np.clip(alpha_lo / alpha_expect, 0., 1.)
            alpha_hi = np.clip(
                (alpha_hi - alpha_expect) / (1. - alpha_expect), 0., 1.)
            # Push to zero from inside if small enough (default is below 7
            # sigma). This avoids having numerics blow up the limits too large
            alpha_lo[np.isclose(alpha_lo, 0., atol=atol)] = 0.
            alpha_hi[np.isclose(alpha_hi, 1., atol=atol)] = 1.
            # Put lo border on a normal log scale and hi ones on an inversed log
            # scale which zooms to values towards 1: [0, 1)->[0, +inf).
            log_alpha_lo = np.log10(alpha_lo)
            log_alpha_hi_inv = -1. * np.log10(1. - alpha_hi)

        # Put log(0) back from inf to the clipping value
        _m_lo = np.isinf(log_alpha_lo)
        _m_hi = np.isinf(log_alpha_hi_inv)
        log_alpha_lo[_m_lo] = np.sign(log_alpha_lo[_m_lo]) * CLIP_LOG
        log_alpha_hi_inv[_m_hi] = (
            -1. * np.sign(log_alpha_hi_inv[_m_hi]) * CLIP_LOG)

        # Next masks sets invalid intervals (Lo==Hi and nans) invisible (0, 0)
        _invalid = np.logical_or(
            np.logical_or(np.isnan(log_alpha_lo), np.isnan(log_alpha_hi_inv)),
            np.isclose(alpha_lo, alpha_hi))
        log_alpha_lo[_invalid] = 0.
        log_alpha_hi_inv[_invalid] = 0.

        # Plot intervalls: Fill between interval borders in scaled axis. Los are
        # filled from -inf to 0 and His from 0 to inf.
        _loga_lo = np.append(log_alpha_lo[0], log_alpha_lo)
        _loga_hi = np.append(log_alpha_hi_inv[0], log_alpha_hi_inv)
        _zeros = np.zeros_like(_loga_lo)
        ax.fill_between(bins, _loga_lo, _zeros, step="pre", color=colors[i])
        ax.fill_between(bins, _zeros, _loga_hi, step="pre", color=colors[i])

        # If we have a single interval we're good (used in count plot later)
        valid_intervals = np.logical_or(valid_intervals, ~_invalid)

        # Store max valid range for plot ylims
        max_log_range = max(
            max_log_range, max(np.amax(np.abs(log_alpha_lo[~_m_lo])),
                               np.amax(log_alpha_hi_inv[~_m_hi])))

    max_log_range = (np.round(max_log_range, decimals=1) + 0.1) * 1.1

    # Plot center line in 'w' or 'k' depending on inner interval color lightness
    _, lightness, _ = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(colors[-1]))
    _c = "k"
    if lightness < 0.35:
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
    # Replace shortcuts to avoid double settings (there is a mpl method for
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

    # No data point is plotted when there isn't a valid interval
    alpha_expect = alpha_expect[valid_intervals]
    alpha_cnts = np.clip(alpha_cnts[valid_intervals], 0., 1.)
    mids = 0.5 * (bins[1:] + bins[:-1])[valid_intervals]

    # Case 1/2: Counts alpha >= alpha_mu
    m_hi = (alpha_cnts >= alpha_expect)
    # Transform to inverted log vals
    alpha_cnts_hi = alpha_cnts[m_hi]
    alpha_cnts_hi = (
        alpha_cnts_hi - alpha_expect[m_hi]) / (1. - alpha_expect[m_hi])
    log_alpha_cnts_hi_inv = -1. * np.log10(1. - alpha_cnts_hi)
    # Outside range? Plot with 'marker_hi' at upper bound, else use 'marker'
    _m = (log_alpha_cnts_hi_inv > max_log_range)
    ax.plot(mids[m_hi][~_m], log_alpha_cnts_hi_inv[~_m],
            marker=markers["marker"], **plot_kwargs)
    ax.plot(mids[m_hi][_m], np.sum(_m) * [max_log_range],
            marker=markers["marker_hi"], **plot_kwargs)

    # Case 2/2: Counts alpha < alpha_mu
    m_lo = (alpha_cnts < alpha_expect)
    # Transform to log vals
    alpha_cnts_lo = alpha_cnts[m_lo]
    alpha_cnts_lo = alpha_cnts_lo / alpha_expect[m_lo]
    log_alpha_cnts_lo_inv = np.log10(alpha_cnts_lo)
    # Outside range? Plot with 'marker_lo' at lower bound
    _m = (log_alpha_cnts_lo_inv < -max_log_range)
    ax.plot(mids[m_lo][_m], np.sum(_m) * [-max_log_range],
            marker=markers["marker_lo"], **plot_kwargs)
    # Zero counts but in range? Use 'marker_zero' at alpha, else use 'marker'
    _m0 = np.logical_and(~_m, counts[valid_intervals][m_lo] == 0)
    ax.plot(mids[m_lo][_m0], log_alpha_cnts_lo_inv[_m0],
            marker=markers["marker_zero"], **plot_kwargs)
    _mrest = np.logical_and(~_m, counts[valid_intervals][m_lo] > 0)
    ax.plot(mids[m_lo][_mrest], log_alpha_cnts_lo_inv[_mrest],
            marker=markers["marker"], **plot_kwargs)

    # #########################################################################
    # Plot settings
    # Build custom ticks in logspace
    def ticker(x, pos):
        """ Make 10^-log(abs(x)) ticks. x is tick value, pos its position """
        if x == 0:
            return "$1$"
        return "$10^{{-{:.0f}}}$".format(np.abs(x))

    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ticker))

    # Labels and range
    ax.set_ylabel(r"$\alpha / \alpha_{{\mu}} \enspace\leftrightarrow\enspace "
                  r"p / \alpha_{{\mu}}$", rotation=90)
    ax.set_xlim(bins[0], bins[-1])
    # 1.05 makes under-/overflow markers fully visible at the edge (trial&error)
    ax.set_ylim(-max_log_range * 1.05, max_log_range * 1.05)

    return ax, alpha_cnts, alpha_expect, alpha_intervals
