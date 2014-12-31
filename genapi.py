#!/usr/bin/env python
"""Generates Cyclus API bindings.
"""
from __future__ import print_function, unicode_literals

import io
import os
import sys
import imp
import json
import argparse
import platform
import warnings
import subprocess
from glob import glob
from distutils import core, dir_util
from pprint import pprint, pformat
if sys.version_info[0] > 2:
    from urllib.request import urlopen
    str_types = (str, bytes)
else:
    from urllib2 import urlopen
    str_types = (str, unicode)

import jinja2

#
# Type System
#

class TypeSystem(object):
    """A type system for cyclus code generation."""

    def __init__(self, table, cycver, cpp_typesystem='cpp_typesystem'):
        """Parameters
        ----------
        table : list
            A table of possible types. The first row must be the column names.
        cycver : tuple of ints
            Cyclus version number.
        cpp_typesystem : str, optional
            The namespace of the C++ wrapper header.

        Attributes
        ----------
        table : list
            A stripped down table of type information.
        cols : dict
            Maps column names to column number in table.
        cycver : tuple of ints
            Cyclus version number.
        verstr : str
            A version string of the format 'vX.X'.
        types : set of str
            The type names in the type system.
        ids : dict
            Maps types to integer identifier.
        cpptypes : dict
            Maps types to C++ type.
        ranks : dict
            Maps types to shape rank.
        norms : dict
            Maps types to programatic normal form, ie INT -> 'int' and
            VECTOR_STRING -> ('std::vector', 'std::string').
        dbtypes : list of str
            The type names in the type system, sorted by id.
        """
        self.cpp_typesystem = cpp_typesystem
        self.cycver = cycver
        self.verstr = verstr = 'v{0}.{1}'.format(*cycver)
        self.cols = cols = {x: i for i, x in enumerate(table[0])}
        id, name, version = cols['id'], cols['name'], cols['version']
        cpptype, rank = cols['C++ type'], cols['shape rank']
        self.table = table = [row for row in table if row[version] == verstr]
        self.types = types = set()
        self.ids = ids = {}
        self.cpptypes = cpptypes = {}
        self.ranks = ranks = {}
        for row in table:
            t = row[name]
            types.add(t)
            ids[t] = row[id]
            cpptypes[t] = row[cpptype]
            ranks[t] = row[rank]
        self.norms = {t: parse_template(c) for t, c in cpptypes.items()}
        self.dbtypes = sorted(types, key=lambda t: ids[t])

        # caches
        self._cython_cpp_name = {}
        self._cython_types = dict(CYTHON_TYPES)
        self._funcnames = dict(FUNCNAMES)
        self._vars_to_py = dict(VARS_TO_PY)
        self._to_py_converters = dict(TO_PY_CONVERTERS)

    def cython_cpp_name(self, t):
        """Returns the C++ name of the type, eg INT -> cpp_typesystem.INT."""
        if t not in self._cython_cpp_name:
            self._cython_cpp_name[t] = '{0}.{1}'.format(self.cpp_typesystem, t)
        return self._cython_cpp_name[t]

    def cython_type(self, t):
        """Returns the Cython spelling of the type."""
        if t in self._cython_types:
            return self._cython_types[t]
        if isinstance(t, str_types):
            n = self.norms[t]
            return self.cython_type(n)
        # must be teplate type
        cyt = list(map(self.cython_type, t))
        cyt = '{0}[{1}]'.format(cyt[0], ', '.join(cyt[1:]))
        self._cython_types[t] = cyt
        return cyt

    def funcname(self, t):
        """Returns a version of the type name suitable for use in a function name.
        """
        if t in self._funcnames:
            return self._funcnames[t]
        if isinstance(t, str_types):
            n = self.norms[t]
            return self.funcname(n)
        f = '_'.join(map(self.funcname, t))
        self._funcnames[t] = f
        return f

    def var_to_py(self, x, t):
        """Returns an expression for converting an object to Python."""
        n = self.norms[t]
        expr = self._vars_to_py.get(n, None)
        if expr is None:
            f = self.funcname(t)
            expr = f + '_to_py({var})'
            self._vars_to_py[n] = expr
        return expr.format(var=x)

    def hold_any_to_py(self, x, t):
        """Returns an expression for converting a hold_any object to Python."""
        cyt = self.cython_type(t)
        cast = '{0}.cast[{1}]()'.format(x, cyt)
        return self.var_to_py(cast, t)

    def convert_to_py(self, x, t):
        """Converts a C++ variable to python.

        Parameters
        ----------
        x : str
            variable name
        t : str
            variable type

        Returns
        -------
        decl : str
            Declarations needed for conversion, may be many lines.
        body : str
            Actual conversion implementation, may be many lines.
        rtn : str
            Return expression.
        """
        n = self.norms.get(t, t)
        ctx = {'type': self.cython_type(t), 'var': x}
        if n in self._to_py_converters:
            # basic type or str
            decl, body, expr = self._to_py_converters[n]
        else:
            # must be a template already
            n0 = n[0]
            decl, body, expr = self._to_py_converters[n0]
            for targ, n_i in zip(TEMPLATE_ARGS[n0], n[1:]):
                x_i = x + '_' +  targ
                ctx[targ+'name'] = x_i
                ctx[targ+'type'] = self.cython_type(n_i)
                dbe_i = self.convert_to_py(x_i, n_i)
                dbe_i = map(Indenter, dbe_i)
                ctx[targ+'decl'], ctx[targ+'body'], ctx[targ+'expr'] = dbe_i
        decl = decl.format(**ctx)
        body = body.format(**ctx)
        expr = expr.format(**ctx)
        return decl, body, expr


CYTHON_TYPES = {
    # type system types
    'BOOL': 'cpp_bool',
    'INT': 'int',
    'FLOAT': 'float',
    'DOUBLE': 'double',
    'STRING': 'std_string',
    'VL_STRING': 'std_string',
    'BLOB': 'cpp_cyclus.Blob',
    'UUID': 'cpp_cyclus.uuid',
    # C++ normal types
    'bool': 'cpp_bool',
    'int': 'int',
    'float': 'float',
    'double': 'double',
    'std::string': 'std_string',
    'std::string': 'std_string',
    'cyclus::Blob': 'cpp_cyclus.Blob',
    'boost::uuids::uuid': 'cpp_cyclus.uuid',
    # Template Types
    'std::set': 'std_set',
    'std::map': 'std_map',
    'std::pair': 'std_pair',
    'std::list': 'std_list',
    'std::vector': 'std_vector',
    }

FUNCNAMES = {
    # type system types
    'BOOL': 'bool',
    'INT': 'int',
    'FLOAT': 'float',
    'DOUBLE': 'double',
    'STRING': 'std_string',
    'VL_STRING': 'std_string',
    'BLOB': 'blob',
    'UUID': 'uuid',
    # C++ normal types
    'bool': 'bool',
    'int': 'int',
    'float': 'float',
    'double': 'double',
    'std::string': 'std_string',
    'std::string': 'std_string',
    'cyclus::Blob': 'blob',
    'boost::uuids::uuid': 'uuid',
    # Template Types
    'std::set': 'std_set',
    'std::map': 'std_map',
    'std::pair': 'std_pair',
    'std::list': 'std_list',
    'std::vector': 'std_vector',
    }

# note that this maps normal forms to python
VARS_TO_PY = {
    'bool': '{var}',
    'int': '{var}',
    'float': '{var}',
    'double': '{var}',
    'std::string': '{var}',
    'cyclus::Blob': 'blob_to_bytes({var})',
    'boost::uuids::uuid': 'uuid_to_py({var})',
    }

TEMPLATE_ARGS = {
    'std::set': ('val',),
    'std::map': ('key', 'val'),
    'std::pair': ('first', 'second'),
    'std::list': ('val',),
    'std::vector': ('val',),
    }

# note that this maps normal forms to python
TO_PY_CONVERTERS = {
    # base types
    'bool': ('', '', '{var}'),
    'int': ('', '', '{var}'),
    'float': ('', '', '{var}'),
    'double': ('', '', '{var}'),
    'std::string': ('', '', '{var}'),
    'cyclus::Blob': ('', '', 'blob_to_bytes({var})'),
    'boost::uuids::uuid': ('', '', 'uuid_to_py({var})'),
    # templates
    'std::set': (
        '{valdecl}'
        'cdef {valtype} {valname}\n'
        'cdef std_set[{valtype}].iterator it\n'
        'cdef set py{var}\n',
        'it = {var}.begin()\n'
        'while it != {var}.end():\n'
        '    {valname} = deref(it)\n'
        '    {valbody.indent4}\n'
        '    pyval = {valexpr}\n'
        '    pyvar.add(pyval)\n'
        '    inc(it)\n',
        'py{var}'),
    'std::map': ('', '', '{var}'),
    'std::pair': ('', '', '{var}'),
    'std::list': ('', '', '{var}'),
    'std::vector': ('', '', '{var}'),
    }

def split_template_args(s, open_brace='<', close_brace='>', separator=','):
    """Takes a string with template specialization and returns a list
    of the argument values as strings. Mostly cribbed from xdress.
    """
    targs = []
    ns = s.split(open_brace, 1)[-1].rsplit(close_brace, 1)[0].split(separator)
    count = 0
    targ_name = ''
    for n in ns:
        count += int(open_brace in n)
        count -= int(close_brace in n)
        targ_name += n
        if count == 0:
            targs.append(targ_name.strip())
            targ_name = ''
    return targs


def parse_template(s, open_brace='<', close_brace='>', separator=','):
    """Takes a string -- which may represent a template specialization --
    and returns the corresponding type. Mostly cribbed from xdress.
    """
    if open_brace not in s and close_brace not in s:
        return s
    t = [s.split(open_brace, 1)[0]]
    targs = split_template_args(s, open_brace=open_brace,
                                close_brace=close_brace, separator=separator)
    for targ in targs:
        t.append(parse_template(targ, open_brace=open_brace,
                                close_brace=close_brace, separator=separator))
    t = tuple(t)
    return t


class Indenter(object):

    def __init__(self, s):
        self._s = s

    def __str__(self):
        return self._s

    def __getattr__(self, key):
        if key.startswith('indent'):
            n = int(key[6:])
            return self._s.replace('\n', '\n' + ' '*n)
        return self.__dict__[key]

#
# Code Generation
#

JENV = jinja2.Environment(undefined=jinja2.StrictUndefined)

CG_WARNING = """
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! WARNING - THIS FILE HAS BEEN !!!!!!
# !!!!!   AUTOGENERATED BY CYMETRIC  !!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
""".strip()

STL_CIMPORTS = """
# Cython standard library imports
from libcpp.map cimport std_map
from libcpp.set cimport std_set
from libcpp.list cimport std_list
from libcpp.vector cimport std_vector
from libcpp.utility cimport std_pair
from libcpp.string cimport string as std_string
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp cimport bool as cpp_bool
""".strip()

NPY_IMPORTS = """
# numpy imports & startup
cimport numpy as np
import numpy as np
np.import_array()
np.import_ufunc()
""".strip()

CPP_TYPESYSTEM = JENV.from_string("""
{{ cg_warning }}

cdef extern from "cyclus.h" namespace "cyclus":

    cdef enum DbTypes:
        {{ dbtypes | join('\n') | indent(8) }}

""".strip())

def cpp_typesystem(ts, ns):
    """Creates the Cython header that wraps the Cyclus type system."""
    ctx = dict(
        dbtypes=ts.dbtypes,
        cg_warning=CG_WARNING,
        stl_cimports=STL_CIMPORTS,
        )
    rtn = CPP_TYPESYSTEM.render(ctx)
    return rtn
    

TYPESYSTEM_PYX = JENV.from_string('''
{{ cg_warning }}

{{ stl_cimports }}

{{ npy_imports }}

# local imports
from cymetric cimport cpp_typesystem
from cymetric cimport cpp_cyclus

# raw type definitions
{% for t in dbtypes %}
{{ t }} = {{ ts.cython_cpp_name(t) }}
{%- endfor -%}

# converters
cdef bytes blob_to_bytes(cpp_cyclus.Blob value):
    rtn = value.str()
    return bytes(rtn)


cdef object uuid_to_py(cpp_cyclus.uuid x):
    cdef int i
    cdef list d = []
    for i in range(16):
        d.append(<unsigned int> x.data[i])
    rtn = uuid.UUID(hex=hexlify(bytearray(d)))
    return rtn


{% for t in dbtypes %}
{% set decl, body, expr = ts.convert_to_py('x', t) %}
cdef object {{ ts.funcname(t) }}_to_py({{ ts.cython_type(t) }} x):
    {{ decl | indent(4) }}
    {{ body | indent(4) }}
    return {{ expr }}
{%- endfor -%}

# type system functions

cdef object db_to_py(cpp_cyclus.hold_any value, cpp_cyclus.DbTypes dbtype):
    """Converts database types to python objects."""
    cdef object rtn
    if dbtype == {{ ts.cython_cpp_name(dbtypes[0]) }}:
        rtn = {{ ts.hold_any_to_py('value', dbtypes[0]) }}
    {%- for t in dbtypes[1:] %}
    elif dbtype == {{ ts.cython_cpp_name(t) }}:
        rtn = {{ ts.hold_any_to_py('value', t) }}
    {%- endfor -%}
    else:
        raise TypeError("dbtype {0} could not be found".format(dbtype))
    return rtn



'''.strip())

def typesystem_pyx(ts, ns):
    """Creates the Cython wrapper for the Cyclus type system."""
    ctx = dict(
        ts=ts,
        dbtypes=ts.dbtypes,
        cg_warning=CG_WARNING,
        npy_imports=NPY_IMPORTS,
        stl_cimports=STL_CIMPORTS,
        )
    rtn = TYPESYSTEM_PYX.render(ctx)
    return rtn


#
# CLI
#

DBTYPES_JS_URL = 'http://fuelcycle.org/arche/dbtypes.js'

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', default='cymetric', dest='src_dir',
                        help="the local source directory, default 'cymetric'")
    parser.add_argument('--test-dir', default='tests', dest='test_dir',
                        help="the local tests directory, default 'tests'")
    parser.add_argument('--build-dir', default='build', dest='build_dir',
                        help="the local build directory, default 'build'")
    parser.add_argument('--cpp-typesystem', default='cpp_typesystem.pxd', 
                        dest='cpp_typesystem',
                        help="the name of the C++ typesystem wrapper header, "
                             "default 'cpp_typesystem.pxd'")
    parser.add_argument('--typesystem-pyx', default='typesystem.pyx', 
                        dest='typesystem_pyx',
                        help="the name of the C++ typesystem wrapper, "
                             "default 'typesystem.pyx'")

    ns = parser.parse_args(argv)
    return ns


def setup(ns):
    # load raw table
    dbtypes_json = os.path.join(ns.build_dir, 'dbtypes.json')
    if not os.path.exists(ns.build_dir):
        os.mkdir(ns.build_dir)
    if not os.path.isfile(dbtypes_json):
        print('Downloading ' + DBTYPES_JS_URL + ' ...')
        f = urlopen(DBTYPES_JS_URL)
        raw = f.read()
        parts = [p for p in raw.split("'") if p.startswith('[')]
        with io.open(dbtypes_json, 'w') as f:
            f.write('\n'.join(parts))
    with io.open(dbtypes_json, 'r') as f:
        tab = json.load(f)
    # get cyclus version
    verstr = subprocess.check_output(['cyclus', '--version'])
    ver = tuple(map(int, verstr.split()[2].split('.')))
    # make and return a type system
    ts = TypeSystem(table=tab, cycver=ver, 
            cpp_typesystem=os.path.splitext(ns.cpp_typesystem)[0])
    #print(ts.norms)
    return ts


def code_gen(ts, ns):
    """Generates code given a type system and a namespace."""
    cases = [(cpp_typesystem, ns.cpp_typesystem), 
             (typesystem_pyx, ns.typesystem_pyx),]
    for func, basename in cases:
        s = func(ts, ns)
        fname = os.path.join(ns.src_dir, basename)
        with io.open(fname, 'w') as f:
            f.write(s)


def main(argv=sys.argv[1:]):
    ns = parse_args(argv)
    ts = setup(ns)
    code_gen(ts, ns)


if __name__ == "__main__":
    main()
