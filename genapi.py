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
else:
    from urllib2 import urlopen

DBTYPES_JS_URL = 'http://fuelcycle.org/arche/dbtypes.js'

class TypeSystem(object):
    """A type system for cyclus code generation."""

    def __init__(self, table, cycver):
        """Parameters
        ----------
        table : list
            A table of possible types.
        cycver : tuple of ints
            Cyclus version number.
        """
        self.cycver = cycver
        self.verstr = verstr = 'v{0}.{1}'.format(*cycver)
        self.cols = cols = {x: i for i, x in enumerate(table[0])}
        vercol = cols['version']
        self.table = [row for row in table if row[vercol] == verstr]
    

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', default='cymetric', dest='src_dir',
                        help="the local source directory, default 'cymetric'")
    parser.add_argument('--test-dir', default='tests', dest='test_dir',
                        help="the local tests directory, default 'tests'")
    parser.add_argument('--build-dir', default='build', dest='build_dir',
                        help="the local build directory, default 'build'")

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
    ts = TypeSystem(table=tab, cycver=ver)
    return ts


def main(argv=sys.argv[1:]):
    ns = parse_args(argv)
    setup(ns)


if __name__ == "__main__":
    main()
