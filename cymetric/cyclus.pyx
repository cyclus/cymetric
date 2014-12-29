"""Python wrapper for cyclus."""
from __future__ import division, unicode_literals

# Cython imports
from libcpp.utility cimport pair as std_pair
from libcpp.set cimport set as std_set
from libcpp.map cimport map as std_map
from libcpp.vector cimport vector as std_vector
from libcpp.string cimport string as std_string
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.stdlib cimport malloc, free
from libcpp cimport bool as cpp_bool

from binascii import hexlify
import uuid

cimport numpy as np
import numpy as np
import pandas as pd

# local imports
from cymetric cimport cpp_cyclus

BOOL = cpp_cyclus.BOOL
INT = cpp_cyclus.INT
FLOAT = cpp_cyclus.FLOAT
DOUBLE = cpp_cyclus.DOUBLE
STRING = cpp_cyclus.STRING
VL_STRING = cpp_cyclus.VL_STRING
BLOB = cpp_cyclus.BLOB
UUID = cpp_cyclus.UUID

cdef bytes blob_to_bytes(cpp_cyclus.Blob value):
    rtn = value.str()
    return bytes(rtn)

cdef object db_to_py(cpp_cyclus.hold_any value, cpp_cyclus.DbTypes dbtype):
    """Converts database types to python objects."""
    cdef int i
    cdef object rtn
    if dbtype == cpp_cyclus.BOOL:
        rtn = value.cast[cpp_bool]()
    elif dbtype == cpp_cyclus.INT:
        rtn = value.cast[int]()
    elif dbtype == cpp_cyclus.FLOAT:
        rtn = value.cast[float]()
    elif dbtype == cpp_cyclus.DOUBLE:
        rtn = value.cast[double]()
    elif dbtype == cpp_cyclus.STRING:
        rtn = value.cast[std_string]()
    elif dbtype == cpp_cyclus.VL_STRING:
        rtn = value.cast[std_string]()
    elif dbtype == cpp_cyclus.BLOB:
        rtn = blob_to_bytes(value.cast[cpp_cyclus.Blob]())
    elif dbtype == cpp_cyclus.UUID:
        d = []
        for i in range(16):
            d.append(<unsigned int> value.cast[cpp_cyclus.uuid]().data[i])
        rtn = hexlify(bytearray(d))
        #rtn = uuid.UUID(hex=hexlify(bytearray(d)))
    else:
        raise TypeError("dbtype {0} could not be found".format(dbtype))
    return rtn


cdef object db_to_dt(cpp_cyclus.DbTypes dbtype):
    """Converts database types to numpty dtypes."""
    cdef object rtn
    if dbtype == cpp_cyclus.BOOL:
        rtn = b'b'
    elif dbtype == cpp_cyclus.INT:
        rtn = b'i4'
    elif dbtype == cpp_cyclus.FLOAT:
        rtn = b'f4'
    elif dbtype == cpp_cyclus.DOUBLE:
        rtn = b'f8'
    elif dbtype == cpp_cyclus.STRING:
        rtn = b'O'
    elif dbtype == cpp_cyclus.VL_STRING:
        rtn = b'O'
    elif dbtype == cpp_cyclus.BLOB:
        rtn = b'O'
    elif dbtype == cpp_cyclus.UUID:
        rtn = b'S16'
    else:
        rtn = b'O'
    return rtn


cdef np.ndarray db_to_array(object values, cpp_cyclus.DbTypes dbtype):
    """Converts values to numpy array."""
    cdef np.ndarray rtn
    cdef str dt = db_to_dt(dbtype)
    if dbtype == cpp_cyclus.BOOL:
        rtn = np.array(values, dtype=dt)
    elif dbtype == cpp_cyclus.INT:
        rtn = np.array(values, dtype=dt)
    elif dbtype == cpp_cyclus.FLOAT:
        rtn = np.array(values, dtype=dt)
    elif dbtype == cpp_cyclus.DOUBLE:
        rtn = np.array(values, dtype=dt)
    elif dbtype == cpp_cyclus.STRING:
        rtn = np.array(values)
    elif dbtype == cpp_cyclus.VL_STRING:
        rtn = np.array(values, dtype=dt)
    elif dbtype == cpp_cyclus.BLOB:
        rtn = np.array(values, dtype=dt)
    elif dbtype == cpp_cyclus.UUID:
        rtn = np.array([v.bytes for v in values], dtype=dt)
    else:
        rtn = np.array(values, dtype=dt)
    return rtn


cdef cpp_cyclus.hold_any py_to_any(object value, cpp_cyclus.DbTypes dbtype):
    """Converts python object to database type in a hold_any instance."""
    cdef int i
    cdef cpp_cyclus.hold_any rtn
    cdef cpp_cyclus.uuid u
    if dbtype == cpp_cyclus.BOOL:
        rtn = rtn.assign[cpp_bool](<bint> value)
    elif dbtype == cpp_cyclus.INT:
        rtn = rtn.assign[int](<int> value)
    elif dbtype == cpp_cyclus.FLOAT:
        rtn = rtn.assign[float](<float>  value)
    elif dbtype == cpp_cyclus.DOUBLE:
        rtn = rtn.assign[double](<double> value)
    elif dbtype == cpp_cyclus.STRING:
        rtn = rtn.assign[std_string](std_string(<const char*> value))
    elif dbtype == cpp_cyclus.VL_STRING:
        rtn = rtn.assign[std_string](std_string(<const char*> value))
    elif dbtype == cpp_cyclus.BLOB:
        rtn = rtn.assign[cpp_cyclus.Blob](cpp_cyclus.Blob(value))
    elif dbtype == cpp_cyclus.UUID:
        for i in range(16):
            u.data[i] = value[i]
        rtn = rtn.assign[cpp_cyclus.uuid](u)
    else:
        raise TypeError("dbtype {0} could not be found".format(dbtype))
    return rtn


cdef class _FullBackend:

    def __cinit__(self):
        """Full backend C++ constructor"""

    def __dealloc__(self):
        """Full backend C++ destructor."""
        #del self.ptx  # don't know why this doesn't work
        free(self.ptx)

    def query(self, std_string table, conds=None):
        """Queries a database table.

        Parameters
        ----------
        table : str
            The table name.
        conds : iterable, optional
            A list of conditions.

        Returns
        -------
        results : pd.DataFrame
            Pandas DataFrame the represents the table
        """
        cdef int i, j
        cdef int nrows, ncols
        cdef std_string field
        cdef cpp_cyclus.QueryResult qr
        cdef std_vector[cpp_cyclus.Cond] cpp_conds
        cdef std_vector[cpp_cyclus.Cond]* conds_ptx
        cdef std_map[std_string, cpp_cyclus.DbTypes] coltypes
        # set up the conditions
        if conds is None:
            conds_ptx = NULL
        else:
            coltypes = (<cpp_cyclus.FullBackend*> self.ptx).ColumnTypes(table)
            for cond in conds:
                field = std_string(<const char*> cond[0])
                cpp_conds.push_back(cpp_cyclus.Cond(field, cond[1], 
                    py_to_any(cond[2], coltypes[field])))
            conds_ptx = &cpp_conds
        # query, convert, and return
        qr = (<cpp_cyclus.FullBackend*> self.ptx).Query(table, conds_ptx)
        nrows = qr.rows.size()
        ncols = qr.fields.size()
        cdef dict res = {j: [] for j in range(ncols)}
        for i in range(nrows):
            for j in range(ncols):
                res[j].append(db_to_py(qr.rows[i][j], qr.types[j]))
        res = {qr.fields[j]: db_to_array(v, qr.types[j]) for j, v in res.items()}
        results = pd.DataFrame(res, columns=[qr.fields[j] for j in range(ncols)])
        return results


class FullBackend(_FullBackend, object):
    """Full backend cyclus database interface."""


cdef class _SqliteBack(_FullBackend):

    def __cinit__(self, std_string path):
        """Full backend C++ constructor"""
        self.ptx = new cpp_cyclus.SqliteBack(path)

    def flush(self):
        """Flushes the database to disk."""
        (<cpp_cyclus.SqliteBack*> self.ptx).Flush()

    def name(self):
        """Retuns the name of the database."""
        return (<cpp_cyclus.SqliteBack*> self.ptx).Name()


class SqliteBack(_SqliteBack, FullBackend):
    """SQLite backend cyclus database interface."""


cdef class _Hdf5Back(_FullBackend):

    def __cinit__(self, std_string path):
        """Full backend C++ constructor"""
        self.ptx = new cpp_cyclus.Hdf5Back(path)

    def flush(self):
        """Flushes the database to disk."""
        (<cpp_cyclus.Hdf5Back*> self.ptx).Flush()

    def name(self):
        """Retuns the name of the database."""
        return (<cpp_cyclus.Hdf5Back*> self.ptx).Name()


class Hdf5Back(_Hdf5Back, FullBackend):
    """HDF5 backend cyclus database interface."""
