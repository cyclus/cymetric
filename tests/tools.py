"""Tools for cymetric tests"""
import os
import subprocess
from functools import wraps

from cymetric import cyclus

DBS = [('test.h5', cyclus.Hdf5Back), ('test.sqlite', cyclus.SqliteBack)]

def setup():
    for fname, _ in DBS:
        if os.path.isfile(fname):
            continue
        subprocess.check_call(['cyclus', '-o' + fname, 'test-input.xml'])


def dbtest(f):
    @wraps(f)
    def wrapper():
        for fname, backend in DBS:
            db = backend(fname)
            yield f, db, fname, backend
    return wrapper

