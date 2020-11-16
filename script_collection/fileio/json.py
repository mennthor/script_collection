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
    _ints = (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64)
    _floats = (np.float_, np.float16, np.float32, np.float64)
    _bools = (np.bool_)
    _arrays = (np.ndarray,)

    def default(self, obj):
        if isinstance(obj, self._ints):
            return int(obj)
        elif isinstance(obj, self._floats):
            return float(obj)
        elif isinstance(obj, self._bools):
            return bool(obj)
        elif isinstance(obj, self._arrays):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


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
