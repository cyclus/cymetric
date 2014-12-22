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

# local imports
from cymetric cimport cpp_cyclus

cdef bytes blob_to_bytes(cpp_cyclus.Blob value):
    rtn = value.str()
    return bytes(rtn)

cdef object db_to_py(cpp_cyclus.hold_any value, cpp_cyclus.DbTypes dbtype):
    """Converts database types to python objects."""
    cdef object rtn
    cdef c
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
    #elif dbtype == cpp_cyclus.BLOB:
    #    rtn = blob_to_bytes(value.cast[cpp_cyclus.Blob]())
    elif dbtype == cpp_cyclus.UUID:
        rtn = value.cast[std_string]()
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
        results : table 
            Rows from the table 
        """
        cdef int i, j
        cdef int nrows, ncols
        cdef cpp_cyclus.QueryResult qr
        cdef std_vector[cpp_cyclus.Cond]* cpp_conds
        if conds is None:
            cpp_conds = NULL
        qr = (<cpp_cyclus.FullBackend*> self.ptx).Query(table, cpp_conds)
        nrows = qr.rows.size()
        ncols = qr.fields.size()
        results = []
        for i in range(nrows):
            row = []
            for j in range(ncols):
                row.append(db_to_py(qr.rows[i][j], qr.types[j]))
            results.append(tuple(row))
        return results


class FullBackend(_FullBackend, object):
    """Full backend cyclus database interface."""


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
    """Full backend cyclus database interface."""
