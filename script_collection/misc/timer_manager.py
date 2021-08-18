import datetime


class TimerManager():
    """
    Timer class for convenient tracking of rough execution times.
    Can be stopped and started manually or by using a context manager.

    Examples
    --------
    ```
    import time

    tmanager = TimerManager()

    # Manually
    t = tmanager.start(name="sleep a bit")
    time.sleep(0.5)
    t.stop()  # Or alternatively: tmanager.stop(name="sleep a bit")

    # Context manager
    with tmanager(name="in a context", verb=True) as t:
        time.sleep(1.5)

    # Show summary of timers
    print(tmanager)
    ```
    """

    class _SingleTimer():
        """ Helper class to store the state of a single timer """

        def __init__(self, name, verb, outer):
            self._t0 = datetime.datetime.utcnow()
            self._t1 = None
            self._name = name
            self._verb = verb
            self._outer = outer  # The outer class reference

        # User methods
        def is_running(self):
            return self._t1 is None

        def is_stopped(self):
            return not self.is_running()

        def is_verb(self):
            return self._verb

        def stop(self):
            # Let the outer class know this timer gets stopped
            self._outer.stop(self._name)

        @property
        def elapsed(self):
            if self.is_running():
                raise RuntimeError("Timer still running")
            return self._t0 - self._t1

        # Context manager
        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            self.stop()

        def _stop(self):
            # Do not overwrite the original stop time if called multiple times.
            # Is called from the outer class to handle list maintanance.
            if self.is_running():
                self._t1 = datetime.datetime.utcnow()

        # Repr
        def __repr__(self):
            return "_SingleTimer(name={})".format(
                "None" if self._name is None else "'" + self._name + "'")

        def __str__(self):
            """ Compiles a summary string for this timer """
            msg = ("Global timer" if self._name is None
                   else "Timer '{}'".format(self._name))
            msg += "\n - Start  : {}".format(self._t0)
            if self._t1 is None:
                msg += "\n - still running"
                return msg
            msg += "\n - Stop   : {}".format(self._t1)
            msg += "\n - Elapsed: {}".format(self._t1 - self._t0)
            return msg

    def __init__(self):
        # Tracks all created and used named timers in their called order
        self._timers = {}
        self._timer_names_ordered = []

    def __call__(self, name=None, verb=False):
        """ See `Timer.start()` """
        return self.start(name, verb)

    def __repr__(self):
        reprs = []
        for name in [None] + self._timer_names_ordered:
            try:
                reprs.append(self._timers[name].__repr__())
            except KeyError:
                pass
        return "TimerManager: [{}]".format(", ".join(reprs))

    def _check_name_avail(self, name):
        if name not in self._timers:
            err = ("No global timer" if name is None
                   else "No timer '{}'".format(name))
            raise KeyError("{} available.".format(err))

    # User methods
    def reset_all(self):
        """ Resets the manager and deletes all timers """
        self.__init__()

    def start(self, name=None, verb=False):
        """
        Start a timer.

        Parameters
        ----------
        name : None or str, optional (default: None)
            If `None`, the global, unnamed timer is started, else a new named
            timer is started.
        verb : bool, optional (default: False)
            If `True` prints a summary after the timer is stopped.

        Raises
        ------
        KeyError: If a named timer with the given name already exists.
        RuntimeError: If the global, unnamed timer is currently running.
        """
        # Input and runtime checks
        if name is None:
            if None in self._timers and self._timers[name].is_running():
                raise RuntimeError("Global timer already running. Stop first.")
        else:
            if not isinstance(name, str):
                raise KeyError("Please use string based timer names.")
            if name in self._timers:
                raise KeyError("Timer name '{}' already used.".format(name))
            self._timer_names_ordered.append(name)

        # Start a new timer and store it with the given name
        self._timers[name] = self._SingleTimer(name, verb, self)

        return self._timers[name]

    def stop(self, name=None):
        """
        Stop a timer.

        Parameters
        ----------
        name : None or str, optional (default: None)
            If `None`, the global, unnamed timer is started, else a new named
            timer is started.

        Raises
        ------
        KeyError: If a timer with the given name (or the global one) is not
                  running.
        RuntimeError: If the requested timer is already stopped.
        """
        self._check_name_avail(name)
        timer = self._timers[name]
        timer._stop()
        if timer.is_verb():
            print(timer)

    def print_stats(self, name=None):
        """
        Prints a stats message string for the given timer name.

        Parameters
        ----------
        name : None or str, optional (default: None)
            Use `None` for the the global, unnamed timer.
        """
        self._check_name_avail(name)
        print(str(self._timers[name]))

    def print_summary(self):
        """ Prints a summery of all timers """
        print(self.__repr__())
        for name in [None] + self._timer_names_ordered:
            try:
                self.print_stats(name)
            except KeyError:
                pass

    def get_timer_list(self):
        """ Returns a list of used named timers in the called order. """
        return self._timer_names_ordered
