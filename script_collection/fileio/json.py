import json
import numpy as np


class NumpyTypesEncoder(json.JSONEncoder):
    """
    Custon JSON encoder to convert numpy types to default JSON encodable types.

    Example
    -------
    Use `json.dump` as usual, but provide the `cls=NumpyTypesEncoder` argument:
    ```
    import json
    import numpy
    from script_collection.fileio.json import NumpyTypesEncoder

    data = {"2d_arr": numpy.array([[1, 2, 4], [1, 3, 9], [1, 5, 25])}

    with open("./ndarray.json", "w") as outfile:
        json.dump(data, outfile, cls=NumpyTypesEncoder)
    ```

    Note
    ----
    References and fame:
    - https://stackoverflow.com/questions/50916422 and
    - https://docs.python.org/3/library/json.html#json.JSONEncoder.default
    """

    def __init__(self, *args, **kwargs):
        types = {k.lower(): t for k, t in np.core.numerictypes.allTypes.items()
                 if isinstance(k, str)}
        self._int = tuple(t for k, t in types.items() if "int" in k)
        self._float = tuple(t for k, t in types.items()
                            if "float" in k and "c" not in k)
        self._complex = tuple(
            t for k, t in types.items()
            if ("complex" in k or k[0] == "c") and "char" not in k)
        self._bool = tuple(t for k, t in types.items() if "bool" in k)
        self._array = (np.ndarray, )
        super().__init__(*args, **kwargs)

    def default(self, obj):
        # In order I think they appear most often
        if isinstance(obj, self._array):
            return obj.tolist()  # Automagically handles all list entries too
        elif isinstance(obj, self._float):
            return float(obj)
        elif isinstance(obj, self._int):
            return int(obj)
        elif isinstance(obj, self._bool):
            return bool(obj)
        elif isinstance(obj, self._complex):
            return "{}+{}j".format(obj.real, obj.imag)

        return super().default(obj)


def serialize_ndarrays(d):
    """
    Traverse through iterable object ``d`` and convert all occuring ndarrays
    to lists to make it JSON serializable.

    Note
    ----
    Better use the `NumpyTypesEncoder` class. This method does not handle the
    numpy data types within the arrays (eg. np.int cannot be serialized out of
    the box).

    Parameters
    ----------
    d : iterable
        Can be dict, list, set, tuple or frozenset.

    Returns
    -------
    d : iterable
        Same as input, but all ndarrays replaced by lists.
    """
    def dict_handler(d):
        return d.items()

    handlers = {list: enumerate, tuple: enumerate,
                set: enumerate, frozenset: enumerate,
                dict: dict_handler}

    def serialize(o):
        for typ, handler in handlers.items():
            if isinstance(o, typ):
                for key, val in handler(o):
                    if isinstance(val, np.ndarray):
                        o[key] = val.tolist()
                    else:
                        o[key] = serialize_ndarrays(o[key])
        return o

    return serialize(d)
