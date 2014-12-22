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


cdef class _FullBackend:

    def __cinit__(self):
        """Full backend C++ constructor"""

    def __dealloc__(self):
        """Full backend C++ destructor."""
        #del self.ptx  # don't know why this doesn't work
        free(self.ptx)

    def query(table, conds=None):
        """Queries a database table.

        Parameters
        ----------
        table : str
            The table name.
        conds : iterable, optional
            A list of conditions.
        """


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


class Hdf5Backend(_Hdf5Back, FullBackend):
    """Full backend cyclus database interface."""
