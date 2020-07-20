# coding: utf-8

"""
Collection of plotting functions for astro plots like skymaps etc.
"""

from .colors import dg, ps_cmap
from ..healpy.coords import DecRaToThetaPhi
from ..algorithms.math import wrap_angle

import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.gridspec as _grs
import matplotlib.ticker as _tic
import healpy as _hp
import scipy.ndimage


class skymap(object):
    """
    skymap
    ======

    Class providing functions to create equatorial skymaps using matplotlib.
    Usage:

        >>> skymap = anapymods.plotting.skymap()
        >>> fig, ax = skymap.figure(fig, gs, proj, **kwargs)
        >>> x, y = skymap.EquCoordsToMapCoords(ra, dec)
        >>> ax.plot(x, y)

    Note
    ----

    Equatorial coordinates only are supported at the moment. The choice of the
    label doesn't change the coordinate system.
    """
    _xtick_defaults = _np.arange(-180, 180 + 30, 30)
    _ytick_defaults = _np.arange(-90, 90 + 30, 30)

    dg = "#262626"  # dark grey

    def __init__(self):
        pass

    def figure(self, fig=None, gs=None, proj="mollweide", **kwargs):
        """
        Creates a new figure and axes with the given projection.

        Parameters
        ----------
        fig : matplotlib.figure instance
            Figure in which the new skymap axes shall be embedded.
            If None (default) , a new figure will be created.
        gs : matplotlib.gridspec.SubplotSpec instance
            The subplot spec for which the new skymap axis is created.
            For example::

                import matplotlib.gridspec as gridspec
                gs_sub = gridspec.GridSpec(nrows=1, ncols=2)[0]
                skymap(gs=gs_sub, ...)

            will create a skymap axis in the left subfigure of the gridspec.
            If None (default) , a new figure, with a single subplot is created.
        proj: String
            Map projection. Can be either "mollweide" (default) or "hammer".

        Other Parameters
        ----------------
        grid: Bool
            Turns the custom grid on or off. (default: True)
        gal_plane : bool
            Wether to plot the galactic plane in the equatorial plot
        label: String
            Label placed at the bottom right showing which coordinate system
            is used. (default: "Equatorial")
        figsize : tuple
            Size of the new figure, only active, if fig=None. (default: 8, 6)
        tex : bool
            If we want to use latex to render the labels. (default: False)
        xticks / yticks : array
            Where to place the x and y ticks and the gridlines (if grid=True).
            Values in degree, must be within the map borders
            [-180 x 180]deg x [-90 x 90]deg. Only yticks get labels. xlabel
            are set manually left and right.
            Note: yticks in range [-10°, 10°], >+85° and <-85° get removed
                  because they interfere with other labels or titles.
            (default xticks: range(-180, 180 + 30, 30))
            (default yticks: range(-90, 90 + 30, 30))
        grid_move_to_back : bool
            If True, the zorder of the grid is set to the very back.
            (default: False)
        gal_plane_label : string or None
            Label for the galactic plane entry in a possible legend. If ``None``
            no entry in the legend is made. (default: ``None``)

        Returns
        -------
        figure: matplotlib figure
            The given figure or the newly created one, if fig was None.
        ax: matplotlib axes
            Skymap axis belonging to the returned figure.
        """

        # Catch kwargs
        grid = kwargs.pop("grid", True)
        gal_plane = kwargs.pop("gal_plane", False)
        label = kwargs.pop("label", "Equatorial")
        figsize = kwargs.pop('figsize', (8, 6))
        tex = kwargs.pop('tex', True)
        xticks = kwargs.pop('xticks', self._xtick_defaults)
        yticks = kwargs.pop('yticks', self._ytick_defaults)
        mtb = kwargs.pop('grid_move_to_back', False)
        gal_plane_label = kwargs.pop('gal_plane_label', None)
        for key in kwargs.keys():
            print("Skipping kwarg '{}'. Has no effect here.".format(key))

        # Check if projection is hammer or mollweide
        # Total valid projections in mpl are:
        # [‘aitoff’, ‘hammer’, ‘lambert’, ‘mollweide’, ‘polar’, ‘rectilinear’]
        if proj.lower() not in ["hammer", "mollweide"]:
            raise ValueError("Projection {0!s} not known. "
                             + "Must be either 'hammer' or 'mollweide'."
                             .format(proj))

        # Create a new figure if None is given
        if fig is None:
            fig = _plt.figure(figsize=figsize)

        # If no GridSpec.SubplotSpec is given, create a new single subplot
        if gs is None:
            gs = _grs.GridSpec(nrows=1, ncols=1)[0]

        # Create the new skymap axis object on the available fig with given gs
        self._ax = fig.add_subplot(gs, projection=proj)

        # Add plot label
        sublabel = proj.lower()
        if tex:
            label = r"\textbf{" + label + r"}"
            sublabel = r"\textit{" + sublabel + r"}"
        self._ax.text(
            x=0.9,
            y=0.0,
            s=label + "\n" + sublabel,
            verticalalignment="bottom",
            horizontalalignment="left",
            transform=self._ax.transAxes,
        )

        # Set the custom grid if necessary. Here an automatic coordinate
        # transformation could be added according to which coordinate system
        # is used.
        if grid:
            self._set_grid(
                xticks=xticks,
                yticks=yticks,
                tex=tex,
                move_to_back=mtb)

        # Add line where the galactic plane is
        if gal_plane:
            self._add_gal_plane(c=dg, lw=1.0, ls="-", label=gal_plane_label)

        return fig, self._ax

    def EquCoordsToMapCoords(self, ra, dec):
        """
        Converts equatorial ra, dec coordinates (in radians) to matplotlib
        map coordinates x, y

        Just a wrapper around the external function to provide a closed usage
        of the skymap class.

        Parameters
        ----------
        ra, dec: array-like
            Right ascension and declination values in radians.

        Returns
        -------
        x, y: array-like
            Corresponding map coordinates, displaying right ascension
            counterclockwise from right to left.
        """
        return EquCoordsToMapCoords(ra, dec)

    def _set_grid(self, xticks, yticks, tex, move_to_back):
        """
        Add a custom grid for the SkyMap to the given axis. Even labelling
        for declination/zenith, but just left and right labels for right
        ascension.

        Be careful: The map projections range from lon[-180, 180] and
        lat[-90, 90]. Coordinates have to be properly transformed.

        Most of this code is from skylab.

        Parameters
        ----------
        x/yticks : array-like
            Tick positions in degrees. Only yticks get labels.
        tex : bool
            Wether to use tex in typesetting the labels.
        move_to_back : bool
            If True, plot grid behind the axis (default: False).

        """
        # Shorthand. Don't want to write self._ax the whole time
        ax = self._ax

        # Plot grid behind everything else
        if move_to_back:
            ax.set_axisbelow(True)

        # Functions for ylabels, xlabels get set manually below
        def y_string(y, pos):
            return r"{0:+.0f}$^\circ$".format(y * 180. / _np.pi)

        # Function restrains grid at x° (can be useful for better visibility)
        ax.set_longitude_grid_ends(90.)
        # Shift xlabel text manually a little outwards
        xmargin = _np.pi / 45.
        # Add latitude (right-asc.) label left/right only to have a tidy map
        left_label = r"24$\,$h"
        right_label = r"0$\,$h"
        if tex:
            left_label = r"\textbf{" + left_label + "}"
            right_label = r"\textbf{" + right_label + "}"
        ax.text(x=-_np.pi - xmargin,  # left label
                y=0.,
                s=left_label,
                size="medium",
                weight="bold",  # only has effect with usetex=True in rcParams
                horizontalalignment="right",
                verticalalignment="center",
                )
        ax.text(x=_np.pi + xmargin,  # right label
                y=0.,
                s=right_label,
                size="medium",
                weight="bold",  # only has effect with usetex=True in rcParams
                horizontalalignment="left",
                verticalalignment="center",
                )

        # Where to show ticks and convert to radians
        xticks = _np.radians(xticks)
        # Remove anything close to 0 as it clutters the plot
        yticks = _np.append(yticks[yticks < -10], yticks[yticks > +10])
        # Remove anything close to +/-90 as it infers with titles
        mask = _np.logical_and(yticks > -85, yticks < +85)
        yticks = yticks[mask]
        yticks = _np.radians(yticks)

        # No major x-tics because we want no xtick labels in the map.
        ax.xaxis.set_major_locator(_tic.NullLocator())
        # Major y-ticks set the label at locations previously defined
        ax.yaxis.set_major_locator(_tic.FixedLocator(yticks))
        # Minor ticks don't show labels but define the grid here.
        ax.xaxis.set_minor_locator(_tic.FixedLocator(xticks))
        ax.yaxis.set_minor_locator(_tic.FixedLocator(yticks))

        # Format y-labels
        ax.yaxis.set_major_formatter(_tic.FuncFormatter(y_string))

        # Only show minor grid, because major is reserved for labels only
        ax.grid(True, which="minor", ls=":", c=dg)

        # Plot the dec = 0° line for better orientation
        zorder = ax.get_zorder() + 1
        ax.plot([-_np.pi, _np.pi], [0, 0], ls="-",
                c=dg, alpha=0.75, zorder=zorder)

        return

    def _add_gal_plane(self, c, lw, ls, label):
        """
        Add the galactic plane in equatorial coordinates to given axis.
        Wrapper for external method only.

        Parameters
        ----------
        ax : matplotlib.axes instance
            Axes to plot the galactic plane on.
        c : matplotlib.color
            Color of the plane line in any matplotlib valid format.
        lw : float
            Linewidth of the plane line.
        ls : matplotlib.linestyle
            Linestyle of the plane line in any matplotlib valid format.
        """
        self._ax = add_gal_plane(self._ax, c, lw, ls, label)
        return


class skymap_local(object):
    """
    skymap
    ======

    Class providing functions to create equatorial skymaps using matplotlib.
    Usage:

        >>> skymap = anapymods.plotting.skymap()
        >>> fig, ax = skymap.figure(fig, gs, proj, **kwargs)
        >>> x, y = skymap.EquCoordsToMapCoords(ra, dec)
        >>> ax.plot(x, y)

    Note
    ----

    Equatorial coordinates only are supported at the moment. The choice of the
    label doesn't change the coordinate system.
    """
    _xtick_defaults = _np.arange(-180, 180 + 30, 30)
    _ytick_defaults = _np.arange(-90, 90 + 30, 30)

    dg = "#262626"  # dark grey

    def __init__(self):
        pass

    def figure(self, fig=None, gs=None, proj="mollweide", **kwargs):
        """
        Creates a new figure and axes with the given projection.

        Parameters
        ----------
        fig : matplotlib.figure instance
            Figure in which the new skymap axes shall be embedded.
            If None (default) , a new figure will be created.
        gs : matplotlib.gridspec.SubplotSpec instance
            The subplot spec for which the new skymap axis is created.
            For example::

                import matplotlib.gridspec as gridspec
                gs_sub = gridspec.GridSpec(nrows=1, ncols=2)[0]
                skymap(gs=gs_sub, ...)

            will create a skymap axis in the left subfigure of the gridspec.
            If None (default) , a new figure, with a single subplot is created.
        proj: String
            Map projection. Can be either "mollweide" (default) or "hammer".

        Other Parameters
        ----------------
        grid: Bool
            Turns the custom grid on or off. (default: True)
        figsize : tuple
            Size of the new figure, only active, if fig=None. (default: 8, 6)
        tex : bool
            If we want to use latex to render the labels. (default: False)
        xticks / yticks : array
            Where to place the x and y ticks and the gridlines (if grid=True).
            Values in degree, must be within the map borders
            [-180 x 180]deg x [-90 x 90]deg. Only yticks get labels. xlabel
            are set manually left and right.
            Note: yticks in range [-10°, 10°], >+85° and <-85° get removed
                  because they interfere with other labels or titles.
            (default xticks: range(-180, 180 + 30, 30))
            (default yticks: range(-90, 90 + 30, 30))
        grid_move_to_back : bool
            If True, the zorder of the grid is set to the very back.
            (default: False)

        Returns
        -------
        figure: matplotlib figure
            The given figure or the newly created one, if fig was None.
        ax: matplotlib axes
            Skymap axis belonging to the returned figure.
        """

        # Catch kwargs
        label = "Local"
        grid = kwargs.pop("grid", True)
        figsize = kwargs.pop('figsize', (8, 6))
        tex = kwargs.pop('tex', True)
        xticks = kwargs.pop('xticks', self._xtick_defaults)
        yticks = kwargs.pop('yticks', self._ytick_defaults)
        mtb = kwargs.pop('grid_move_to_back', False)
        for key in kwargs.keys():
            print("Skipping kwarg '{}'. Has no effect here.".format(key))

        # Check if projection is hammer or mollweide
        # Total valid projections in mpl are:
        # [‘aitoff’, ‘hammer’, ‘lambert’, ‘mollweide’, ‘polar’, ‘rectilinear’]
        if proj.lower() not in ["hammer", "mollweide"]:
            raise ValueError("Projection {0!s} not known. "
                             + "Must be either 'hammer' or 'mollweide'."
                             .format(proj))

        # Create a new figure if None is given
        if fig is None:
            fig = _plt.figure(figsize=figsize)

        # If no GridSpec.SubplotSpec is given, create a new single subplot
        if gs is None:
            gs = _grs.GridSpec(nrows=1, ncols=1)[0]

        # Create the new skymap axis object on the available fig with given gs
        self._ax = fig.add_subplot(gs, projection=proj)

        # Add plot label
        sublabel = proj.lower()
        if tex:
            label = r"\textbf{" + label + r"}"
            sublabel = r"\textit{" + sublabel + r"}"
        self._ax.text(
            x=0.9,
            y=0.0,
            s=label + "\n" + sublabel,
            verticalalignment="bottom",
            horizontalalignment="left",
            transform=self._ax.transAxes,
        )

        # Set the custom grid if necessary. Here an automatic coordinate
        # transformation could be added according to which coordinate system
        # is used.
        if grid:
            self._set_grid(
                xticks=xticks,
                yticks=yticks,
                tex=tex,
                move_to_back=mtb)

        return fig, self._ax

    def _ThetaPhiToMapCoords(self, theta, phi):
        """
        Just a wrapper around the external function to provide a closed usage
        of the skymap class.

        Transforms local (healpy) theta (zenith), phi (azimuth) values to
        matplotlib map projection values x, y, so that the phi values are shown
        counterclockwise from right (phi=0) to left (phi=2pi) on the map and
        zenith from top (theta=0) to bottom (theta=180).

        Parameters
        ----------
        theta, phi: array-like
            Local healpy theta, phi (zenith, azimuth) coordinates in radians.

        Returns
        -------
        x, y: array-like
            Corresponding map coordinates, displaying phi (azimuth)
            counterclockwise from right to left.
        """
        return ThetaPhiToMapCoords(theta, phi)

    def _set_grid(self, xticks, yticks, tex, move_to_back):
        """
        Add a custom grid for the SkyMap to the given axis. Even labelling
        for zenith, but just left and right labels for azimuth.

        Be careful: The map projections range from lon[-180, 180] and
        lat[-90, 90]. Coordinates have to be properly transformed.

        Most of this code is from skylab.

        Parameters
        ----------
        x/yticks : array-like
            Tick positions in degrees. Only yticks get labels.
        tex : bool
            Wether to use tex in typesetting the labels.
        move_to_back : bool
            If True, plot grid behind the axis (default: False).

        """
        # Shorthand. Don't want to write self._ax the whole time
        ax = self._ax

        # Plot grid behind everything else
        if move_to_back:
            ax.set_axisbelow(True)

        # Functions for ylabels, xlabels get set manually below
        def y_string(y, pos):
            # Shift to 0° (y=90°) zenith at the top and 180°
            # at the bottom (y=-90°)
            converted = (_np.pi / 2. - y) * 180. / _np.pi
            return r"{0:.0f}$^\circ$".format(converted)

        # Function restrains grid at x° (can be useful for better visibility)
        ax.set_longitude_grid_ends(90.)
        # Shift xlabel text manually a little outwards
        xmargin = _np.pi / 45.
        # Add latitude (right-asc.) label left/right only to have a tidy map
        left_label = r"$360^\circ$"
        right_label = r"$0^\circ$"
        if tex:
            left_label = r"\textbf{" + left_label + "}"
            right_label = r"\textbf{" + right_label + "}"
        ax.text(x=-_np.pi - xmargin,  # left label
                y=0.,
                s=left_label,
                size="medium",
                weight="bold",  # only has effect with usetex=True in rcParams
                horizontalalignment="right",
                verticalalignment="center",
                )
        ax.text(x=_np.pi + xmargin,  # right label
                y=0.,
                s=right_label,
                size="medium",
                weight="bold",  # only has effect with usetex=True in rcParams
                horizontalalignment="left",
                verticalalignment="center",
                )

        # Where to show ticks and convert to radians
        xticks = _np.radians(xticks)
        # Remove anything close to 0 as it clutters the plot
        yticks = _np.append(yticks[yticks < -10], yticks[yticks > +10])
        # Remove anything close to +/-90 as it infers with titles
        mask = _np.logical_and(yticks > -85, yticks < +85)
        yticks = yticks[mask]
        yticks = _np.radians(yticks)

        # No major x-tics because we want no xtick labels in the map.
        ax.xaxis.set_major_locator(_tic.NullLocator())
        # Major y-ticks set the label at locations previously defined
        ax.yaxis.set_major_locator(_tic.FixedLocator(yticks))
        # Minor ticks don't show labels but define the grid here.
        ax.xaxis.set_minor_locator(_tic.FixedLocator(xticks))
        ax.yaxis.set_minor_locator(_tic.FixedLocator(yticks))

        # Format y-labels
        ax.yaxis.set_major_formatter(_tic.FuncFormatter(y_string))

        # Only show minor grid, because major is reserved for labels only
        ax.grid(True, which="minor", ls=":", c=dg)

        # Plot the dec = 0° line for better orientation
        zorder = ax.get_zorder() + 1
        ax.plot([-_np.pi, _np.pi], [0, 0], ls="-",
                c=dg, alpha=0.75, zorder=zorder)

        return


class recmap(object):
    """
    Rectilinear Map
    ===============

    Class providing functions to create equatorial maps in rectilinear
    projection using matplotlib. Usage:

        >>> recmap = anapymods.plotting.recmap()
        >>> fig, ax = recmap.figure(fig, gs, **kwargs)
        >>> x, y = recmap.EquCoordsToMapCoords(ra, dec)
        >>> ax.plot(x, y)

    Note
    ----

    Equatorial coordinates only are supported at the moment.
    """
    _xtick_defaults = range(-180, 180 + 30, 30)
    _ytick_defaults = range(-90, 90 + 30, 30)

    def __init__(self):
        pass

    def figure(self, fig=None, gs=None, **kwargs):
        """
        Creates a new figure and axes with the given projection.

        Parameters
        ----------
        fig : matplotlib.figure instance
            Figure in which the new skymap axes shall be embedded.
            If None (default) , a new figure will be created.
        gs : matplotlib.gridspec.SubplotSpec instance
            The subplot spec for which the new skymap axis is created.
            For example::

                import matplotlib.gridspec as gridspec
                gs_sub = gridspec.GridSpec(nrows=1, ncols=2)[0]
                skymap(gs=gs_sub, ...)

            will create a skymap axis in the left subfigure of the gridspec.
            If None (default) , a new figure, with a single subplot is created.

        Other Parameters
        ----------------
        grid: Bool
            Turns the custom grid on or off. (default: True)
        gal_plane : bool
            Wether to plot the galactic plane in the equatorial plot
        figsize : tuple
            Size of the new figure, only active, if fig=None. (default: 8, 6)
        xticks / yticks : array
            Where to place the x and y ticks and the gridlines (if grid=True).
            Values in degree, must be within the map borders and in ascending
            order [-180 x 180]deg x [-90 x 90]deg.
            If range is smaller than full range, the map is zoomed.
        grid_move_to_back : bool
            If True, the zorder of the grid is set to the very back.
            (default: False)

        Returns
        -------
        figure: matplotlib figure
            The given figure or the newly created one, if fig was None.
        ax: matplotlib axes
            Skymap axis belonging to the returned figure.
        """
        # Catch kwargs
        grid = kwargs.pop("grid", True)
        gal_plane = kwargs.pop("gal_plane", False)
        figsize = kwargs.pop('figsize', (8, 6))
        xticks = kwargs.pop('xticks', self._xtick_defaults)
        yticks = kwargs.pop('yticks', self._ytick_defaults)
        mtb = kwargs.pop('grid_move_to_back', False)
        for key in kwargs.keys():
            print("Skipping kwarg '{}'. Has no effect here.".format(key))

        # Create a new figure if None is given
        if fig is None:
            fig = _plt.figure(figsize=figsize)
        # If no GridSpec.SubplotSpec is given, create a new single subplot
        if gs is None:
            gs = _grs.GridSpec(nrows=1, ncols=1)[0]
        # Create the new skymap axis object on the available fig with given gs
        self._ax = fig.add_subplot(gs)

        # Set plot limits
        xlim = _np.deg2rad(xticks)[[0, -1]]
        ylim = _np.deg2rad(yticks)[[0, -1]]

        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)

        self._ax.set_xlabel("RA in h")
        self._ax.set_ylabel("DEC in °")

        # Set the custom grid if wanted
        if grid:
            self._set_grid(
                xticks=xticks,
                yticks=yticks,
                move_to_back=mtb)

        # Add line where the galactic plane is
        if gal_plane:
            self._add_gal_plane(c=dg, lw=1.0, ls="-")

        return fig, self._ax

    def EquCoordsToMapCoords(self, ra, dec):
        """
        Converts equatorial ra, dec coordinates (in radians) to matplotlib
        map coordinates x, y

        Just a wrapper around the external function to provide a closed usage
        of the skymap class.

        Parameters
        ----------
        ra, dec: array-like
            Right ascension and declination values in radians.

        Returns
        -------
        x, y: array-like
            Corresponding map coordinates, displaying right ascension
            counterclockwise from right to left.
        """
        return EquCoordsToMapCoords(ra, dec)

    def _set_grid(self, xticks, yticks, move_to_back):
        """
        Add a custom grid for the given axis.
        Be careful: The map projections range from lon[-180, 180] and
        lat[-90, 90]. Coordinates have to be properly transformed.

        Parameters
        ----------
        x/yticks : array-like
            Tick positions in degrees.
        move_to_back : bool
            If True, plot grid behind the axis (default: False).
        """
        # Shorthand. Don't want to write self._ax the whole time
        ax = self._ax

        # Plot grid behind everything else
        if move_to_back:
            ax.set_axisbelow(True)

        # Functions for x, y labels
        def x_string(x, pos):
            # Convert to hours from right 0h (180°) to left 24h (-180°)
            converted = (_np.pi - x) / (2 * _np.pi) * 24
            if _np.isclose(converted // 2, 0.):
                return r"{0:2.0f}".format(converted)
            else:
                return r"{0:2.1f}".format(converted)

        def y_string(y, pos):
            return r"{0:2.0f}".format(y * 180. / _np.pi)

        # Convert to radians
        xticks = _np.radians(xticks)
        yticks = _np.radians(yticks)

        # Minor ticks don't show labels but define the grid here.
        ax.xaxis.set_major_locator(_tic.FixedLocator(xticks))
        ax.yaxis.set_major_locator(_tic.FixedLocator(yticks))
        # No minor tics, don't need them
        ax.xaxis.set_minor_locator(_tic.NullLocator())
        ax.yaxis.set_minor_locator(_tic.NullLocator())

        # Format y-labels
        ax.xaxis.set_major_formatter(_tic.FuncFormatter(x_string))
        ax.yaxis.set_major_formatter(_tic.FuncFormatter(y_string))

        # Turn major grid on
        ax.grid(which="both", ls=":", c=dg)

        # Plot the dec = 0° line for better orientation
        zorder = ax.get_zorder() + 1
        ax.plot([-_np.pi, _np.pi], [0, 0], ls="-",
                c=dg, alpha=1, zorder=zorder)

        return

    def _add_gal_plane(self, c, lw, ls):
        """
        Add the galactic plane in equatorial coordinates to given axis.
        Wrapper for external method only.

        Parameters
        ----------
        ax : matplotlib.axes instance
            Axes to plot the galactic plane on.
        c : matplotlib.color
            Color of the plane line in any matplotlib valid format.
        lw : float
            Linewidth of the plane line.
        ls : matplotlib.linestyle
            Linestyle of the plane line in any matplotlib valid format.
        """
        self._ax = add_gal_plane(self._ax, c, lw, ls)
        return


class recmap_local(object):
    """
    Rectilinear Map
    ===============

    Class providing functions to create equatorial maps in rectilinear
    projection using matplotlib. Usage:

        >>> recmap = anapymods.plotting.recmap()
        >>> fig, ax = recmap.figure(fig, gs, **kwargs)
        >>> x, y = recmap.EquCoordsToMapCoords(ra, dec)
        >>> ax.plot(x, y)

    Note
    ----

    Local coordinates only are supported at the moment.
    """
    _xtick_defaults = range(-180, 180 + 30, 30)
    _ytick_defaults = range(-90, 90 + 30, 30)

    def __init__(self):
        pass

    def figure(self, fig=None, gs=None, **kwargs):
        """
        Creates a new figure and axes with the given projection.

        Parameters
        ----------
        fig : matplotlib.figure instance
            Figure in which the new skymap axes shall be embedded.
            If None (default) , a new figure will be created.
        gs : matplotlib.gridspec.SubplotSpec instance
            The subplot spec for which the new skymap axis is created.
            For example::

                import matplotlib.gridspec as gridspec
                gs_sub = gridspec.GridSpec(nrows=1, ncols=2)[0]
                skymap(gs=gs_sub, ...)

            will create a skymap axis in the left subfigure of the gridspec.
            If None (default) , a new figure, with a single subplot is created.

        Other Parameters
        ----------------
        grid: Bool
            Turns the custom grid on or off. (default: True)
        figsize : tuple
            Size of the new figure, only active, if fig=None. (default: 8, 6)
        xticks / yticks : array
            Where to place the x and y ticks and the gridlines (if grid=True).
            Values in degree, must be within the map borders and in ascending
            order [-180 x 180]deg x [-90 x 90]deg.
            If range is smaller than full range, the map is zoomed.
        grid_move_to_back : bool
            If True, the zorder of the grid is set to the very back.
            (default: False)

        Returns
        -------
        figure: matplotlib figure
            The given figure or the newly created one, if fig was None.
        ax: matplotlib axes
            Skymap axis belonging to the returned figure.
        """
        # Catch kwargs
        grid = kwargs.pop("grid", True)
        figsize = kwargs.pop('figsize', (8, 6))
        xticks = kwargs.pop('xticks', self._xtick_defaults)
        yticks = kwargs.pop('yticks', self._ytick_defaults)
        mtb = kwargs.pop('grid_move_to_back', False)
        for key in kwargs.keys():
            print("Skipping kwarg '{}'. Has no effect here.".format(key))

        # Create a new figure if None is given
        if fig is None:
            fig = _plt.figure(figsize=figsize)
        # If no GridSpec.SubplotSpec is given, create a new single subplot
        if gs is None:
            gs = _grs.GridSpec(nrows=1, ncols=1)[0]
        # Create the new skymap axis object on the available fig with given gs
        self._ax = fig.add_subplot(gs)

        # Set plot limits
        xlim = _np.deg2rad(xticks)[[0, -1]]
        ylim = _np.deg2rad(yticks)[[0, -1]]

        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)

        # Set the custom grid if wanted
        if grid:
            self._set_grid(
                xticks=xticks,
                yticks=yticks,
                move_to_back=mtb)

        return fig, self._ax

    def _set_grid(self, xticks, yticks, move_to_back):
        """
        Add a custom grid for the given axis.
        Be careful: The map projections range from lon[-180, 180] and
        lat[-90, 90]. Coordinates have to be properly transformed.

        Parameters
        ----------
        x/yticks : array-like
            Tick positions in degrees.
        move_to_back : bool
            If True, plot grid behind the axis (default: False).
        """
        # Shorthand. Don't want to write self._ax the whole time
        ax = self._ax

        # Plot grid behind everything else
        if move_to_back:
            ax.set_axisbelow(True)

        # Functions for x, y labels
        def x_string(x, pos):
            # Convert to degrees from right 0° (x=180°) to left 360° (x=-180°)
            converted = (_np.pi - x) / (2 * _np.pi) * 360
            return r"{0:.0f}".format(converted)

        def y_string(y, pos):
            # Shift to 0° (y=90°) zenith at the top and 180°
            # at the bottom (y=-90°)
            converted = (_np.pi / 2. - y) * 180. / _np.pi
            return r"{0:.0f}".format(converted)

        # Convert to radians
        xticks = _np.radians(xticks)
        yticks = _np.radians(yticks)

        # Minor ticks don't show labels but define the grid here.
        ax.xaxis.set_major_locator(_tic.FixedLocator(xticks))
        ax.yaxis.set_major_locator(_tic.FixedLocator(yticks))
        # No minor tics, don't need them
        ax.xaxis.set_minor_locator(_tic.NullLocator())
        ax.yaxis.set_minor_locator(_tic.NullLocator())

        # Format y-labels
        ax.xaxis.set_major_formatter(_tic.FuncFormatter(x_string))
        ax.yaxis.set_major_formatter(_tic.FuncFormatter(y_string))

        # Turn major grid on
        ax.grid(which="both", ls=":", c=dg)

        # Plot the dec = 0° line for better orientation
        zorder = ax.get_zorder() + 1
        ax.plot([-_np.pi, _np.pi], [0, 0], ls="-",
                c=dg, alpha=1, zorder=zorder)

        return


def add_gal_plane(ax, c=dg, lw=1.0, ls="-", label=None):
    """
    Add the galactic plane in equatorial coordinates to given axis.

    Coordinate transformation from "Practical Astronomy with your
    Calculator", page 58 and additionally
    from https://gist.github.com/barentsen/2367839 and Wikipedia
    "Galactic coordinates".

    Parameters
    ----------
    ax : matplotlib.axes instance
        Axes to plot the galactic plane on.
    c : matplotlib.color
        Color of the plane line in any matplotlib valid format.
    lw : float
        Linewidth of the plane line.
    ls : matplotlib.linestyle
        Linestyle of the plane line in any matplotlib valid format.

    Returns
    -------
    ax : matplotlib.axes instance
        The axis with the plane plotted on.
    """
    # Galactic plane is a line at the horizon in gal. coords (b=height)
    lat = _np.linspace(0., 2 * _np.pi, 360 + 1)
    b = _np.zeros_like(lat)
    # North galactic pole (J2000)
    pole_ra = _np.radians(192.859508)
    pole_dec = _np.radians(27.128336)
    posangle = _np.radians(122.932 - 90.0)
    # Transform from galactic to equatorial coords
    ra = (_np.arctan2(
        _np.cos(b) * _np.cos(lat - posangle),
        (_np.sin(b) * _np.cos(pole_dec)
            - _np.cos(b) * _np.sin(pole_dec) * _np.sin(lat - posangle))
    ) + pole_ra)
    dec = _np.arcsin(_np.cos(b) * _np.cos(pole_dec)
                     * _np.sin(lat - posangle) + _np.sin(b)
                     * _np.sin(pole_dec))
    # Map some outlier ra values back to interval [0, 2pi]
    ra[ra > 2 * _np.pi] -= 2 * _np.pi
    # Transform to map coords
    x, y = EquCoordsToMapCoords(ra, dec)

    # Sort in x to display proper line and fix first point at -180°
    xidx = _np.argsort(x)
    x = _np.append([-_np.pi], x[xidx])
    y = _np.append(y[xidx][0], y[xidx])

    # Plot plane line
    if label is not None:
        ax.plot(x, y, color=c, lw=lw, linestyle=ls, label=label)
    else:
        ax.plot(x, y, color=c, lw=lw, linestyle=ls)

    return ax


def ThetaPhiToMapCoords(theta, phi):
    """
    Transforms local (healpy) theta (zenith), phi (azimuth) values to
    matplotlib map projection values x, y, so that the phi values are shown
    counterclockwise from right (phi=0) to left (phi=2pi) on the map and
    zenith from top (theta=0) to bottom (theta=180).

    The map coordinates are defined in the ranges x:[-pi, pi] and
    y:[-pi/2, pi/2], where (x|y) = (0|0) is the middle of the map and
    (x|y) = (-pi|pi/2) is the top left point on the map (near the upper
    pole).

    Parameters
    ----------
    theta, phi: array-like
        Local healpy theta, phi (zenith, azimuth) coordinates in radians.

    Returns
    -------
    x, y: array-like
        Corresponding map coordinates, displaying phi (azimuth)
        counterclockwise from right to left.
    """
    if _np.any(phi) < 0 or _np.any(phi) > 2 * _np.pi:
        raise ValueError("Phi not in range [0, 2*pi].")
    if _np.any(theta) < 0. or _np.any(theta) > _np.pi:
        raise ValueError("Theta not in range [0, pi].")

    x = _np.pi - phi
    y = _np.pi / 2. - theta
    return x, y


def EquCoordsToMapCoords(ra, dec):
    """
    Transforms equatorial ra, dec values to matplotlib map projection
    values x, y, so that the ra values are shown counterclockwise from
    right (ra=0) to left (ra=2pi) on the map.

    The map coordinates are defined in the ranges x:[-pi, pi] and
    y:[-pi/2, pi/2], where (x|y) = (0|0) is the middle of the map and
    (x|y) = (-pi|pi/2) is the top left point on the map (near the upper
    pole).

    So here we map ra in [0, 2pi] to x in [-pi, pi], so that ra:0 -> x:pi
    and ra:2pi -> x:-pi to plot ra counterclockwise from right
    (ra=0, x=pi) to left (ra=2pi, x=-pi).

    The declination dec is already in the same range and orientation as y
    and need not be converted.

    Healpy pixels have no fixed orientation, they are given in standard
    spherical coordinates theta in [0, pi] and phi in [0, 2pi]. So it must
    be properly understood how sky coordinates are mapped to pixels.
    In the default healpy views, (phi|theta) = (0|0) is the first pixel
    and (pi, 2*pi) the last pixel.
    Attention: The internal healpy map functions show the middle pixel at
               (phi|theta) = (pi/2|pi) at the middle right of the map (so
               just using the standard map coordinates). So healpy map
               views have (theta|phi)=(pi/2|0) at the center and (0|0) in
               the top left.

    Parameters
    ----------
    ra, dec: array-like
        Right ascension and declination values in radians.

    Returns
    -------
    x, y: ndarray
        Corresponding map coordinates, displaying right ascension
        counterclockwise from right to left.
    """
    if _np.any(ra) < 0 or _np.any(ra) > 2 * _np.pi:
        raise ValueError("RA not in range [0, 2*pi].")
    if _np.any(dec) < -_np.pi / 2. or _np.any(dec) > _np.pi / 2.:
        raise ValueError("DEC not in range [-pi/2, +pi/2].")

    x = _np.pi - _np.atleast_1d(ra)
    y = _np.atleast_1d(dec)
    return x, y


def plot_healpy_map(m, ax=None, **kwargs):
    """
    Plot a healpy map on a matplotlib map projection using pcolormesh.

    Parameters
    ----------
    map : healpy map
        Healpy maps are simple arrays where each entry belongs to a pixel of a
        healpy map generated with a specific ordering and resolution. Here RING
        ordering (default) is used and the number of pixels is the length of
        the map and must be a number 12*2**k.
    ax : matplotlib.axis instance
        Axis on which to plot the healpy map.
        If None (default), a new single standard subplot will be generated.

    Other Parameters
    ----------------
    figsize : tuple
            Size of the new figure, only active, if fig=None. (default: 8, 6)
    cbar_label : string
        Label for the colorbar. (default: "")
    cbar_orient : string
        Orientation of the colorbar: None, "horizontal" or  "vertical".
        If None (default), no colorbar is plotted.
    cmap : string
        Named matplotlib colormap. (default: custom ps_cmap is used)
    rasterize : bool
        Wether to rasterize the pcolormesh for performace reasons.
        (default: True)
    renderpix : int
        How many pixel shall be rendered in pcolormesh. The more pixel we
        render the longer it takes, but the finer the detail we see from the
        healpy map are. Rendered are renderpix pixels in x in renderpix / 2
        pixels in y direction. If 0 the pixel are set to match the healpix
        resolution of the given map. (default: 0)
    draw_contour : bool
        If ``True``draw contours instead of a pcolormesh. (default: ``False``)
    levels: array-like or int
        Levels if ``draw_contour``is ``True``. (default: 10)
    extend : string
        Option for contourf. Can be ``'neither', 'min', 'max', 'both'``. This is
        needed to extend the color scale above and / or below the given levels.
        (default: "neither")


    Returns
    -------
    figure: matplotlib figure
        The given figure or the newly created one, if axis was None.
    ax: matplotlib axes
        The given axis or the newly created one, if axis was None.
    """
    # Get kwargs
    figsize = kwargs.pop("figsize", (8, 6))
    cbar_label = kwargs.pop("cbar_label", "")
    cbar_orient = kwargs.pop("cbar_orient", None)
    cmap = kwargs.pop("cmap", ps_cmap)
    rasterize = kwargs.pop("rasterize", True)
    renderpix = kwargs.pop("renderpix", 0)
    draw_contour = kwargs.pop("draw_contour", False)
    levels = kwargs.pop("levels", 10)
    extend = kwargs.pop("extend", "neither")

    for key in kwargs.keys():
        print("Skipping kwarg '{}'. Has no effect here.".format(key))

    # If no axis is given, create a new single subplot
    if ax is None:
        fig = _plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    else:  # else use the existing one
        fig = ax.figure

    # Generate healpy parameters from the given map
    NSIDE = _hp.get_nside(m)

    # Generate a grid in healpy coords phi and theta.
    # They are defined as azimuth and zenith, phi:[0,2pi], theta:[0,pi]
    # The first pixel is at (0,0) the last one at (2pi,pi).
    # N defines the number of rectangles drawn by pcolormesh and sets the
    # pixel resolution of the map. Note: this resolution is independent of the
    # healpy map resolution and affects only how much pixels are drawn on the
    # matplotlib map. N should be large enough to get a finer resolution than
    # the healpy map.
    if renderpix == 0:
        # Render only as many pixels, as the current figure can display
        renderpix = _np.amax(fig.get_dpi() * fig.get_size_inches())
    renderpix = int(_np.ceil(renderpix))
    # Create the outer gridpoints
    p = _np.linspace(0, 2 * _np.pi, renderpix + 1)
    t = _np.linspace(0, _np.pi, int(_np.ceil(renderpix / 2)) + 1)

    # Create the meshgrid for pcolormesh in healpy coords
    pp, tt = _np.meshgrid(p, t)

    # ## Most important part: Store the right map value
    # ## for every pixel on the grid at the right location
    # ## in a new 2D z array.
    # pix are the pix indices of the grid points
    # if the res of the healpy map is small, many grid pixels get the same
    # pix index and we the the typical healpy map structure
    pix = _hp.ang2pix(NSIDE, tt, pp)
    z = m[pix]

    # Now we have to map the healpy grid to the map coords. We must only
    # shift lineary otherwise the grid gets destroyed.
    # ## BE CAREFUL: DO NOT WRAP ANGLES
    # ## The grid will be destroyed and the coordinates get messed up
    # Map coordinates ra=0 right to left, dec=90 top to bottom
    dec = _np.pi / 2. - tt
    ra = pp
    xx, yy = EquCoordsToMapCoords(ra, dec)
    # Equivalent to:
    xx = _np.pi - pp
    yy = _np.pi / 2. - tt

    # Create the pcolormesh plot. The returned mesh is used for the colorbar.
    if not draw_contour:
        mesh = ax.pcolormesh(xx, yy, z, shading="flat", cmap=cmap,
                             rasterized=rasterize)
    else:
        mesh = ax.contourf(xx, yy, z, levels, cmap=cmap, extend=extend)

    # Create a colorbar if desired
    if cbar_orient is not None:
        cbar = fig.colorbar(
            mesh,
            orientation=cbar_orient,
            fraction=0.05,
            shrink=0.8,
            aspect=25,
        )
        cbar.set_label(cbar_label)

    return fig, ax


def mollview(m, **kwargs):
    """
    Wrapper around `plot_healpy_map()` to provide a function similar to
    `healpy.mollview()`.

    Other Parameters
    ----------------
    coords : string
        Defines the coordinate system of the map. Can be "equatorial" or
        "local" (default).
    cmap = matplotlib.colormap
        Which colormap to use in plotting
    """
    coords = kwargs.pop("coords", "local")
    cmap = kwargs.pop("cmap", ps_cmap)
    for arg in kwargs.keys():
        raise ValueError("Don't know args : '{}'".format(arg))

    if coords == "local":
        sm = skymap_local()
    elif coords == "equatorial":
        sm = skymap()
    else:
        raise ValueError("'coords' must be either 'local' or 'equatorial'.")
    fig, ax = sm.figure(tex=False)

    # Render only as much pixels as the current figure can display in its
    # largest dimension.
    renderpix = _np.amax(fig.get_dpi() * fig.get_size_inches())
    _, ax = plot_healpy_map(m, ax=ax, renderpix=renderpix,
                            cbar_orient="horizontal", cmap=cmap)

    return fig, ax


def cartview(m, **kwargs):
    """
    Wrapper around `plot_healpy_map()` to provide a function similar to
    `healpy.cartview()`.

    Other Parameters
    ----------------
    coords : string
        Defines the coordinate system of the map. Can be "equatorial" or
        "local" (default).
    cmap = matplotlib.colormap
        Which colormap to use in plotting
    """
    coords = kwargs.pop("coords", "local")
    cmap = kwargs.pop("cmap", ps_cmap)

    if coords == "local":
        rm = recmap_local()
    elif coords == "equatorial":
        rm = recmap()
    else:
        raise ValueError("'coords' must be either 'local' or 'equatorial'.")
    fig, ax = rm.figure(figsize=(8, 4))

    # Render only as much pixels as the current figure can display in its
    # largest dimension.
    renderpix = _np.amax(fig.get_dpi() * fig.get_size_inches())
    _, ax = plot_healpy_map(m, ax=ax, renderpix=renderpix,
                            cbar_orient="vertical", cmap=cmap)

    ax.set_xlim(-_np.pi, _np.pi)
    ax.set_ylim(-_np.pi / 2., _np.pi / 2.)

    if coords == "local":
        ax.set_xlabel("azimuth in deg")
        ax.set_ylabel("zenith in deg")
    else:
        ax.set_xlabel("right-ascension in h")
        ax.set_ylabel("declination in deg")

    ax.grid(ls=":", c=dg)

    return fig, ax


def mollview_grid(maps, **kwargs):
    """
    Plots a grid of mollview skymaps with a healpy map each.

    Parameters
    ----------
    maps : 3d array
        axis0 = rows, axis1 = columns, axis2 = maps

        Example: maps.shape = (1, 2, 3145728) produces 1 row and 2 colmuns and
                 plots two maps with resolution 512 side by side.

    Other Parameters
    ----------------
    labels : array
        String array with labels to be plotted on the map in the same shape
        as the first two map dimensions: maps.shape[:2].
    projections : array
        String array with projections to used on the maps in the same shape
        as the first two map dimensions: maps.shape[:2].
        Projections can be "mollweide" (default) or "hammer".
    renderpix : int
        The number of pixels that shall be to used to render each map.
        If 0, the resolution is set automatically dependent on the map
        resolution. (default: 0)
    Returns
    -------
    fig : matplotlib.figure instance
        The newly created figure.
    ax : matplotlib.axes instance
        The newly created subplot axis, in th esame shape as the first two
        maps dimensions.
    """
    # Sanity checks
    maps = _np.array(maps)
    mapshape = _np.array(maps.shape)
    if len(mapshape) != 3:
        raise ValueError("Maps must be in a 3d array.")
    if _np.argmax(mapshape) != 2:
        raise ValueError("maps must be stored in the last axis=2."
                         "Or choose a smaller grid < len(map).")

    # Pop kwargs
    labels = _np.array(kwargs.pop(
        "labels", ["Equatorial"] * mapshape[0] * mapshape[1]))
    projections = _np.array(kwargs.pop(
        "projections", ["mollweide"] * mapshape[0] * mapshape[1]))
    rpix = kwargs.pop("renderpix", 0)

    # Create gridspec matching the map array
    gs = _grs.GridSpec(nrows=mapshape[0], ncols=mapshape[1])

    # Get skymap object
    sm = skymap()
    # Enlarge figure when using more maps, each map gets 8x8 inches
    fig = _plt.figure(figsize=(mapshape[1] * 4, mapshape[0] * 2))

    # Reshape loop arrays to have each map stacked up in a row
    maps = maps.reshape(mapshape[0] * mapshape[1], mapshape[2])
    labels = labels.reshape(mapshape[0] * mapshape[1])
    projections = projections.reshape(mapshape[0] * mapshape[1])

    ax = []
    for gsi, mi, labeli, proji in zip(gs, maps, labels, projections):
        _, axi = sm.figure(fig=fig, gs=gsi, tex=False,
                           label=labeli, proj=proji)
        ax.append(axi)
        # Plot current map in the current axis
        _, axi = plot_healpy_map(mi, ax=axi, renderpix=rpix)

    # Shape ax as the map array
    ax = _np.array(ax)
    ax_ = ax.reshape(mapshape[:2])

    # Pad horizontally
    fig.tight_layout(w_pad=2, h_pad=2)

    return fig, ax_


# Some new functions taking a different and more direct approach than above
def cartzoom(m, ra_rng=[0, 2 * _np.pi], dec_rng=[-_np.pi / 2., _np.pi / 2.],
             ax=None, **kwargs):
    """
    Zoom in a rectangular region of a map in equatorial coords.

    Parameters
    ----------
    m : array-like
        Valid healpy map.
    ra_rng : tuple
        ``[ra_min, ra_max]`` borders of the zoomed region.
    dec_rng : tuple
        ``[dec_min, dec_max]`` borders of the zoomed region.
    ax : maptlotlib.axes instance, optional
        Axis to draw on. (Default: None)

    Returns
    -------
    fig, ax, img
    """
    ra_rng, dec_rng = map(_np.sort, map(_np.atleast_1d, [ra_rng, dec_rng]))
    if len(ra_rng) != 2 or len(dec_rng) != 2:
        raise ValueError("ra or dec range must have [min, max]")

    if ax is None:
        aspect_max = max(_plt.rcParamsDefault["figure.figsize"])
        figsize = kwargs.pop("figsize", (aspect_max, aspect_max / 2.))
        fig, ax = _plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    m = _np.atleast_1d(m)
    try:
        NSIDE = _hp.get_nside(m)
    except TypeError:
        raise TypeError("Given map is no valid healpy map.")

    npix_x, npix_y = fig.get_dpi() * fig.get_size_inches()

    ras = _np.linspace(*ra_rng, num=int(npix_x))
    decs = _np.linspace(*dec_rng, num=int(npix_y))

    rr, dd = _np.meshgrid(ras, decs)
    RR, DD = map(_np.ravel, [rr, dd])

    TH, PHI = DecRaToThetaPhi(dec=DD, ra=RR)
    zz = m[_hp.ang2pix(nside=NSIDE, theta=TH, phi=PHI)].reshape(dd.shape)

    ax.set_xlim(_np.amin(rr), _np.amax(rr))
    ax.set_ylim(_np.amin(dd), _np.amax(dd))
    ax.set_xlabel("ra")
    ax.set_ylabel("dec")
    img = ax.pcolormesh(rr, dd, zz, **kwargs)

    return fig, ax, img


def cartzoom_radius(m, r=None, center=[0., 0.], ax=None, **kwargs):
    """
    Same as cartzoom, but around a center with given radius r.

    Parameters
    ----------
    m : array-like
        Valid healpy map.
    r : float, optional
        Radius of the zoomed region, if None, whole map is shown.
        (default: ``None``)
    center : tuple, optional
        ``(ra, dec)`` center of the zoomed region. (Default: ``[0, 0]``))
    ax : maptlotlib.axes instance, optional
        Axis to draw on. (Default: None)

    Returns
    -------
    fig, ax, img
    """
    if r is None:
        return cartzoom(m, ax=ax, **kwargs)
    else:
        if r <= 0:
            raise ValueError("r must be positive in radians")
        r2 = min(r, _np.pi)
        ra0, dec0 = center
        scale = _np.cos(dec0)
        ra_rng = _np.clip([ra0 - r2 / scale, ra0 + r2 / scale], 0., 2. * _np.pi)
        dec_rng = _np.clip([dec0 - r2, dec0 + r2], -_np.pi / 2., _np.pi / 2.)
        return cartzoom(m, ra_rng=ra_rng, dec_rng=dec_rng, ax=ax, **kwargs)


def make_astro_xaxis(ax, time_xax=False):
    """
    Format axis in deg or hours, astro style ra from right to left.
    """
    def ra2time(ra):
        time = ra / 2. / _np.pi * 24.
        hours = int(time)
        arcmins = int(_np.round((time - hours) * 60.))
        return hours, arcmins

    # Set ra tics to appropriate and precise intervalls. Also double the range
    # [-2pi, 4pi] for wrapped axis, use modulo for the text labels later
    xlim = _np.array(ax.get_xlim())
    dx = _np.abs(_np.diff(xlim))
    mins = False
    if dx > _np.deg2rad(180):
        xtics = _np.arange(-360, 360 + 360 + 60, 30)
    elif dx > _np.deg2rad(90):
        xtics = _np.arange(-360, 360 + 360 + 30, 30)
    elif dx > _np.deg2rad(45):
        xtics = _np.arange(-360, 360 + 360 + 15, 15)
    elif dx > _np.deg2rad(15):
        xtics = _np.arange(-360, 360 + 360 + 5, 5)
        mins = True
    else:
        xtics = _np.arange(-360, 360 + 360 + 1, 1)
        mins = True
    xtics = _np.deg2rad(xtics)
    xtics = xtics[_np.logical_and(xtics >= xlim[0], xtics <= xlim[1])]
    ax.set_xticks(xtics)

    xlabs = ax.get_xticklabels()
    while len(xlabs) > 0:
        xlabs.remove(xlabs[0])

    # Manually write text
    for tici in xtics:
        if tici != 2. * _np.pi:
            tici = wrap_angle(tici)
        if time_xax:  # Make ra in hours if desired
            if not mins:
                text = "{:2d}".format(ra2time(tici)[0])
            else:
                text = "{:2d}h{:02d}'".format(*ra2time(tici))
        else:
            text = "{:.0f}".format(_np.rad2deg(tici))
        xlabs.append(_plt.Text(tici, 0, text))
    ax.set_xticklabels(xlabs)

    # Set dec to degrees in appropriate precise steps
    ylim = ax.get_ylim()
    dy = _np.abs(_np.diff(ylim))
    if dy > _np.deg2rad(45):
        ytics = _np.arange(-90, 90 + 15, 15)
    elif dy > _np.deg2rad(15):
        ytics = _np.arange(-90, 90 + 5, 5)
    else:
        ytics = _np.arange(-90, 90 + 1, 1)
    ytics = _np.deg2rad(ytics)
    ytics = ytics[_np.logical_and(ytics >= ylim[0], ytics <= ylim[1])]
    ax.set_yticks(ytics)

    ylabs = ax.get_yticklabels()
    while len(ylabs) > 0:
        ylabs.remove(ylabs[0])

    for tici in ytics:
        ylabs.append(_plt.Text(tici, 0, u"{:.0f}°".format(_np.rad2deg(tici))))
    ax.set_yticklabels(ylabs)

    # Reverse xaxis so we start with 0h to the right
    ax.set_xlim(xlim[::-1])

    if time_xax:
        ax.set_xlabel("ra in h")
    else:
        ax.set_xlabel("ra in deg")
    ax.set_ylabel("dec in deg")
    return ax


def reconstruct_cartzoom_data(img, smooth=0.):
    """
    Rewconstruct the data used in pcolormesh used to draw a cartzoom.
    Can be used here to draw contour lines on the same plot.

    Parameters
    ----------
    img : image object from pcolormesh
        Returned from cartzoom for example
    x_rng, y_rng : array-like, shape (2)
        Plot range used in plot that created the ``img``.
    smooth : float
        If ``> 0`` a gaussian image filter with width ``smmoth`` is applied to
        the ``img`` pixel data.  Unit of smooth is pixels so choose depending on
        the ``img`` resolution. Can be useful to smooth contours that are built
        from a healpy map.

    Returns
    -------
    x, y, z : array-like
        x-, y-grid and z data array, all with the same 2D shape.
    """
    # Reconstruct mesh and img array for contour plot. StackOverflow:34840366
    x_rng = img._bbox.intervalx
    y_rng = img._bbox.intervaly
    x_npts, y_npts = img._meshWidth, img._meshHeight
    z = img.get_array().reshape(y_npts, x_npts)

    # Smooth image grid with a gaussian filter (for smooth contours)
    if smooth < 0:
        raise ValueError("Smooth must be a positive float.")

    z = scipy.ndimage.filters.gaussian_filter(z, smooth)

    x = _np.linspace(*x_rng, num=x_npts)
    y = _np.linspace(*y_rng, num=y_npts)
    x, y = _np.meshgrid(x, y)

    return x, y, z


def render_healpymap(
        m, npix=(300, 200), bounds=[[0, 2. * _np.pi], [0, _np.pi]], smooth=0.):
    """
    Renders a healpy map in a rectangular grid for plotting or contours.
    ``cartview`` and ``molvview`` method should use this method internally.
    Deprecates ``reconstruct_cartzoom_data``.

    Parameters
    ----------
    m : array-like
        Healpy map array. Convention: ``Phi`` ist the azimuth and ``theta``
        the zenith angle.
    npix : tuple
        ``(npix_phi, npix_th)`` number of pixels to use for the render.
    bounds : array-like
        ``[[phi_lo, phi_hi], [theta_lo, theta_hi]]`` to constrain the render to
        a subspace of the map. ``0<= phi < 2pi, 0 <= theta <= pi``.
    smooth : float
        If ``> 0`` a gaussian image filter with width ``smooth`` is applied to
        the rendered pixel data.  Unit of smooth is pixels so choose depending
        on the ``img`` resolution. Can be useful to smooth the underlying map
        for building contours from a healpy map.

    Returns
    -------
    render_arr : array-like, shape (npix[0], npix[1])
        The map is rendered, so that the first axis is the y-axis (``theta``)
        and the second axis the x-axis (``phi``) as expected by ``pcolormesh``.
    corners_phi : array-like, shape (npix[0], npix[1])
        x-values as expected by ``pcolormesh``.
    corners_th : array-like, shape (npix[0], npix[1])
        y-values as expected by ``pcolormesh``.
    """
    NSIDE = _hp.get_nside(m)
    b_phi, b_th = _np.atleast_2d(bounds)

    phi_corners = _np.linspace(b_phi[0], b_phi[1], npix[0] + 1)
    th_corners = _np.linspace(b_th[0], b_th[1], npix[1] + 1)
    phi_mids = 0.5 * (phi_corners[:-1] + phi_corners[1:])
    th_mids = 0.5 * (th_corners[:-1] + th_corners[1:])

    xx, yy = map(_np.ravel, _np.meshgrid(phi_mids, th_mids))
    idx = _hp.ang2pix(phi=xx, theta=yy, nside=NSIDE)
    map_vals = m[idx].reshape(len(th_mids), len(phi_mids))

    if smooth < 0:
        raise ValueError("Smooth must be a positive float.")
    else:
        map_vals = scipy.ndimage.filters.gaussian_filter(map_vals, smooth)

    return map_vals, phi_corners, th_corners
