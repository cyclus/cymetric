"""Python wrapper for cyclus."""
from __future__ import division, unicode_literals

# Cython imports
from libcpp.utility cimport pair as cpp_pair
from libcpp.set cimport set as cpp_set
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector as cpp_vector
from libcpp.string cimport string as std_string
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.stdlib cimport malloc, free
from libcpp cimport bool as cpp_bool

# local imports
from cymetric cimport cpp_cyclus

cdef class _QueryResult:

    def __cinit__(self):
       """QueryResult C++ constructor"""


class QueryResult(_QueryResult, object):
    """Results froma database query."""



cdef class _FullBackend:

    def __cinit__(self):
       """Full backend C++ constructor"""

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

