# coding: utf-8

"""
Unsorted helper methods.
"""


import sys
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass
import numpy as np
import scipy.interpolate as sci
from itertools import permutations


def round_mantissa(f, way="ceil"):
    """
    Ceils or floors the mantissa.

    Parameters
    ----------
    f : float or array-like
        Values which mantissas shall get ceiled / floored.
    way : str
        Can be ``ceil`` or ``floor``. (default: ``ceil``)

    Returns
    -------
    f_round : float or array-like
        Values with ceiled / floored mantissas.

    Examples
    --------
    >>> round_mantissa(np.array([0.1, 0.0006, 23.4, -1.5, -449.293]), "ceil")
    array([ 1.e-01,  6.e-04,  3.e+01, -2.e+00, -5.e+02])
    >>> round_mantissa(np.array([0.1, 0.0006, 23.4, -1.5, -449.293]), "floor")
    array([ 1.e-01,  5.e-04,  2.e+01, -1.e+00, -4.e+02])
    """
    sign = np.sign(f)
    abs_f = np.abs(f)
    exp = np.floor(np.log10(abs_f))
    exp10 = 10**exp
    mant = abs_f / exp10
    if way == "ceil":
        way = np.ceil
    elif way == "floor":
        way = np.floor
    else:
        raise ValueError("`way` can be 'ceil' or 'floor'.")
    return sign * way(mant) * exp10


def transpose_list(lst):
    """
    Transposes a list of lists, if each sublist has equal length::

        [[1,2,3],[4,5,6],[7,8,9]] --> [[1,4,7],[2,5,8],[3,6,9]]

    If sublists have unequal length, every list is chopped after the length of
    the shortest list.

    From: https://stackoverflow.com/questions/6473679/transpose-list-of-lists

    Parameters
    ----------
    lst : list of lists
        List of lists to be transposed.

    Returns
    -------
    lT : list of lists
        Transposed lists
    """
    return list(map(list, zip(*lst)))


def total_size(o, handlers={}, verbose=False, full_out=False):
    """
    Returns the approximate memory footprint an object and all of its
    contents by recursively traversing the object tree in bytes.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add ``handlers`` to specify a rule on how to
    iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    Parameters
    ----------
    o : object
    handlers : dict, optional
        Specify content handlers other than the builtins. (default: ``{}``)
    verbose : bool, optional
        If ``True``, print size of each traversed object to ``stderr``.
        (default: ``False``)
    full_out : bool, optional
        If ``True``, return dict with info about each traversed object.
        (default: ``False``)

    Returns
    -------
    total_size : int
        Total size of the object in bytes.
    out : dict, optional
        Only if ``full_out`` is ``True``.
        Dict with keys ``idx``, ``size``, ``type`` and ``repr`` under which each
        traversed object is listed with it's size, type and representation
        information (same as get's printed with ``verbose=True``).

    Note
    ----
    This super useful code is taken from:
        http://code.activestate.com/recipes/577504/
    and modified with the helpful comment from "Sean Worley" to also track
    referenced classes with either ``__dict__`` or ``__slots__``.
    """
    # Builtin object handlers
    def dict_handler(d):
        return chain.from_iterable(d.items())

    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    # Update with user defined handler
    all_handlers.update(handlers)
    # Track which object ids have already been seen
    seen = set()
    # Estimate sizeof object without `__sizeof__`
    default_size = sys.getsizeof(0)

    # Save output
    if full_out:
        out = {"idx": [], "size": [], "type": [], "repr": []}

    def sizeof(o):
        # Do not double count the same object
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=sys.stderr)
        if full_out:
            try:
                out["idx"].append(out["idx"][-1] + 1)
            except (TypeError, IndexError):
                out["idx"].append(0)
            out["size"].append(s)
            out["type"].append(type(o))
            out["repr"].append(repr(o))

        # Recurse to remaining linked objects
        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
            else:  # Expand to classes, comment from 'Sean Worley'
                if not hasattr(o.__class__, '__slots__'):
                    # No __slots__ *usually* means a __dict__, but some
                    # special builtin classes (eg. `type(None)`) have neither
                    if hasattr(o, '__dict__'):
                        s += sizeof(o.__dict__)
                # Else, `o` has no attributes at all, so sys.sys.getsizeof()
                # actually returned the correct value
                else:
                    s += sum(sizeof(getattr(o, x)) for x in
                             o.__class__.__slots__ if hasattr(o, x))
                    # Also add in the __slots__ list, even if it's shared
                    # across all class instances
                    s += sizeof(o.__class__.__slots__)
        return s

    if full_out:
        return sizeof(o), out
    else:
        return sizeof(o)
