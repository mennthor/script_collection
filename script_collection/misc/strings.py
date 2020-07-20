# coding: utf8

"""
Collection of string manipulation and pretty print stuff.
"""

import os
import numpy as np


def float2tex(f, prec=2):
    """
    Returns a string formatted in tex format for scientific exponentials.

    From: https://stackoverflow.com/questions/13490292

    Parameters
    ----------
    f : float
        Float number to be converted to string.
    prec : int, optional
        Desired precision of the float part. (default: 2)

    Returns
    -------
    s : string
        Formated string "x.xx \\times 10^{yyy}".

    Note
    ----
    To use this string in matplotlib: 'r"${}$".format(float2tex(3.1415e5))'

    From `https://docs.python.org/2/library/string.html`_:

    The precise rules are as follows: suppose that the result formatted with
    presentation type ``'e'`` and precision ``p-1`` would have exponent ``exp``.
    Then if ``-4 <= exp < p``, the number is formatted with presentation type
    ``'f'`` and precision ``p-1-exp``. Otherwise, the number is formatted with
    presentation type ``'e'`` and precision ``p-1``. In both cases insignificant
    trailing zeros are removed from the significand, and the decimal point is
    also removed if there are no remaining digits following it.
    """
    float_str = "{:.{:d}g}".format(f, prec + 1)
    if "e" in float_str:
        mantissa, exp = float_str.split("e")
        if mantissa == "1":
            return r"10^{{{}}}".format(int(exp))
        else:
            return r"{} \times 10^{{{}}}".format(mantissa, int(exp))
    else:
        return float_str


def sec2timestr(sec):
    """
    Takes a time in seconds and formats it to:

      ddd : hh : mm : ss.sss

    Parameters
    ----------
    sec : float
        Input time span in seconds.

    Returns
    -------
    timestr : str
        Formatted string.
    """
    raise DeprecationWarning("Use `sec2str` it gives the same result.")
    sec = np.around(sec, decimals=3)
    d = int(sec / (24. * 60. * 60.))
    sec -= d * (24. * 60. * 60.)
    h = int(sec / (60. * 60.))
    sec -= h * (60. * 60.)
    m = int(sec / 60.)
    s = sec - m * 60.
    return "{:d}d : {:02d}h : {:02d}m : {:06.3f}s".format(d, h, m, s)


def sec2str(sec):
    """
    Takes a time in seconds and formats it to:

      ddd : hh : mm : ss.sss

    Note: Potentially more flexible solution than ``sec2timestr``.

    Parameters
    ----------
    sec : float
        Input time span in seconds.

    Returns
    -------
    timestr : str
        Formatted string.
    """
    factors = [24. * 60. * 60., 60. * 60., 60., 1.]
    labels = ["d", "h", "m", "s"]
    splits = []
    for i, factor in enumerate(factors):
        splits.append(sec // factor)
        sec -= splits[-1] * factor

    out = []
    for i, (f, l) in enumerate(zip(splits, labels)):
        # First entry has more digits for overflow
        if i == 0:
            out.append("{:.0f}{}".format(f, l))
        else:
            out.append("{:02.0f}{}".format(f, l))

    return ":".join(out)


def arr2str(arr, sep=", ", fmt="{}"):
    """
    Make a string from a list seperated by ``sep`` and each item formatted
    with ``fmt``.
    """
    return sep.join([fmt.format(v) for v in arr])


def indent_wrap(s, indent=0, wrap=80):
    """
    Wraps and indents a string ``s``.

    Parameters
    ----------
    s : str
        The string to wrap.
    indent : int
        How far to indent each new line.
    wrape : int
        Number of character after which to wrap the string.

    Returns
    -------
    s : str
        Indented and wrapped string, each line has length ``wrap``, except the
        last one, which may have less than ``wrap`` characters.

    Example
    -------
    >>> s = 2 * "abcdefghijklmnopqrstuvwxyz"
    >>> indent_wrap(s, indent=0, wrap=26)
    'abcdefghijklmnopqrstuvwxyz\nabcdefghijklmnopqrstuvwxyz'
    >>> indent_wrap(s, indent=2, wrap=26)
    '  abcdefghijklmnopqrstuvwx\n  yzabcdefghijklmnopqrstuv\n  wxyz'
    """
    split = wrap - indent
    chunks = [indent * " " + s[i:i + split] for i in range(0, len(s), split)]
    return "\n".join(chunks)


def shorten_str(s, flen=3, blen=3, sep="..."):
    """
    Shortens a string by combining the first ``flen`` and last ``blen``
    characters, seperated by ``sep`` if string is longer than
    ``flen + len(sep) + blen``.

    Parameters
    ----------
    s : str
        The string to shorten.
    flen, blen : int, optional
        How many first and last characters to use for the shortened string.
        (default: 3)
    sep : string, optional
        Seperator for the combined, shortended first and last part of the
        string. (default: ``'...'``)

    Returns
    -------
    short_str : str
        The shortened string if ``len(s) > flen + len(sep) + blen``, else ``s``.
    """
    if len(s) > flen + len(sep) + blen:
        s = "{}{}{}".format(s[:flen], sep, s[-blen:] if blen > 0 else "")
    return s


def split_all_ext(fname):
    """
    Splits all extensions from the given filename. You can recombine the
    filename afterwards with ``fname = filename + "".join(extensions)``.

    Parameters
    ----------
    fname : str
        Filename to split.

    Returns
    -------
    filename : str
        Filename without extensions.
    extensions : list of str
        List of all extensions in order from left to right.
    """
    splitext = os.path.splitext(fname)
    extensions = []
    while splitext[1] != "":
        extensions.append(splitext[1])
        splitext = os.path.splitext(splitext[0])
    return splitext[0], extensions[::-1]
