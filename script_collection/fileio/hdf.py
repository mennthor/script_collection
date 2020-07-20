# coding: utf-8

"""
Collection of HDF5 related methods.
"""

import tables


def rec2hdf(rec, h5file, group):
    """
    Writes a record-array to new carrays.

    Parameters
    ----------
    rec : record-array
        The numpy record array to write.
    h5file : tables.File
        The HDF5 file instance open for writing.
    group : tables.Group
        The group to write into (eg. tables.File.create_group("/", "name")).

    Example
    -------
    >>> rec = np.empty((10,), dtype=[("a", float), ("b", float)])
    >>> with tables.open_file("./record.hdf5", mode="w") as h5file:
    >>>     group = h5file.create_group("/", "trials")
    >>>     rec2hdf(rec, h5file, group)
    >>>     h5file.flush()
    """
    # Loop over rec names and create a carray with the same dtype and name
    for name in rec.dtype.names:
        atom = tables.Atom.from_dtype(rec[name].dtype)
        carray = h5file.create_carray(group, name, atom, rec[name].shape)
        carray[:] = rec[name]
    return


# # Not tested!
# def list2hdf(cols, col_names, col_types, h5file, group, table_name):
#     """
#     Write a list of lists to a hdf5 file.

#     Pretty sure this never works...

#     Parameters
#     ----------
#     cols : list of arrays
#         For each array in cols a new column is written.
#     col_names : list
#         How each column shall be named, must be same length as l.
#     col_types : list of tables.Types
#         The type of each list.
#     h5file : tables.File
#         The HDF5 file instance open for writing.
#     group : tables.Group
#         The group to write into (eg. tables.File.create_group("/", "name")).
#     table_name : string
#         How the table is named in the given group.
#
#     Example
#     -------
#     Manually we would do:

#     >>> with tables.open_file("./record.hdf5", mode="w") as h5f:
#     >>>     table = h5f.create_table(h5f.root, "stats",
#     >>>                              {"nzeros": tables.IntCol(),
#     >>>                               "ntrials": tables.IntCol()})
#     >>>     table_row = table.row
#     >>>     table_row["nzeros"] = ntrials - nonzero
#     >>>     table_row["ntrials"] = ntrials
#     >>>     table_row.append()
#     >>>     h5f.flush()
#
#     How is this working, if each column has different lengths?
#     """
#     n_cols = len(cols)
#     n_names = len(col_names)
#     n_types = len(col_types)
#     try:
#         assert n_cols == n_names == n_types
#     except AssertionError:
#         raise ValueError("Length of names, list and types must match.")

#     # Create the table first with the correct shapes
#     shapes = [arr.shape for arr in cols]
#     table = h5f.create_table(group, table_name,
#                              {name: typ(shape=shape) for (name, typ, shape) in
#                               zip(col_names, col_types, shapes)})

#     # Get each column and fill
#     for i, name in enumerate(col_names):
#         eval("table.cols.name = cols[{}]".format(i))
