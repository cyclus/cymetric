"""Tools for cymetric tests"""
import os
import shutil
import subprocess
from functools import wraps

from cymetric import cyclus

DBS = [('test.h5', 'orig.h5', cyclus.Hdf5Back), 
       ('test.sqlite', 'orig.sqlite', cyclus.SqliteBack)]
#DBS = [('test.h5', cyclus.Hdf5Back)]
#DBS = [('test.sqlite', cyclus.SqliteBack)]

def setup():
    for fname, oname, _ in DBS:
        if os.path.isfile(oname):
            continue
        subprocess.check_call(['cyclus', '-o' + oname, 'test-input.xml'])


def dbtest(f):
    @wraps(f)
    def wrapper():
        for fname, oname, backend in DBS:
            if os.path.exists(fname):
                os.remove(fname)
            shutil.copy(oname, fname)
            db = backend(fname)
            yield f, db, fname, backend
    return wrapper

