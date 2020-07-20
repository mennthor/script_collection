# coding: utf-8

"""
Dictionary utility methods.
"""


def dict_map(func, d):
    """
    Applies func to each dict value, returns results in a dict with the same
    keys as the original d.

    Parameters
    ----------
    func : callable
        Function ``func(key, val)`` applied to all ke, value pairs in ``d``.
    d : dict
        Dictionary which values are to be mapped.

    Returns
    -------
    out : dict
        New dict with same key as ``d`` and ``func`` applied to ``d.items()``.
    """
    return {key: func(key, val) for key, val in d.items()}


def dict_filter(func, d):
    """
    Applies func to each dict value, and only write the key, value pair in the
    returned dict, if ``func(val) == True``.

    Parameters
    ----------
    func : callable
        Function ``func(key, val)`` applied to all ke, value pairs in ``d``.
    d : dict
        Dictionary which values are to be mapped.

    Returns
    -------
    out : dict
        New dict with all keys in ``d`` where ``func`` applied to ``d.items()``
        returns ``True``.

    """
    return {key: val for key, val in d.items() if func(key, val)}


def fill_dict_defaults(d, required_keys=None, opt_keys=None, noleft=True):
    """
    Populate dictionary with data from a given dict ``d``, and check if ``d``
    has required and optional keys. Set optionals with default if not present.

    If input ``d`` is None and ``required_keys`` is empty, just return
    ``opt_keys``.

    Parameters
    ----------
    d : dict or None
        Input dictionary containing the data to be checked. If is ``None``, then
        a copy of ``opt_keys`` is returned. If ``opt_keys`` is ``None``, a
        ``TypeError`` is raised. If ``d``is ``None`` and ``required_keys`` is
        not, then a ``ValueError`` israised.
    required_keys : list or None, optional
        Keys that must be present  and set in ``d``. (default: None)
    opt_keys : dict or None, optional
        Keys that are optional. ``opt_keys`` provides optional keys and default
        values ``d`` is filled with if not present in ``d``. (default: None)
    noleft : bool, optional
        If True, raises a ``KeyError``, when ``d`` contains etxra keys, other
        than those given in ``required_keys`` and ``opt_keys``. (default: True)

    Returns
    -------
    out : dict
        Contains all required and optional keys, using default values, where
        optional keys were missing. If ``d`` was None, a copy of ``opt_keys`` is
        returned, if ``opt_keys`` was not ``None``.
    """
    if required_keys is None:
        required_keys = []
    if opt_keys is None:
        opt_keys = {}
    if d is None:
        if not required_keys:
            if opt_keys is None:
                raise TypeError("`d` and òpt_keys` are both None.")
            return opt_keys.copy()
        else:
            raise ValueError("`d` is None, but `required_keys` is not empty.")

    d = d.copy()
    out = {}
    # Set required keys
    for key in required_keys:
        if key in d:
            out[key] = d.pop(key)
        else:
            raise KeyError("Dict is missing required key '{}'.".format(key))
    # Set optional values, if key not given
    for key, val in opt_keys.items():
        out[key] = d.pop(key, val)
    # Complain when extra keys are left and noleft is True
    if d and noleft:
        raise KeyError("Leftover keys ['{}'].".format(
            "', '".join(list(d.keys()))))
    return out
