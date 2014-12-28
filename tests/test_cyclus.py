"""Tests for cyclus wrappers"""
import os
import subprocess
from functools import wraps

import nose
from nose.tools import assert_equal, assert_less

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


@dbtest
def test_name(db, fname, backend):
    obs = db.name()
    assert_equal(fname, obs)


@dbtest
def test_simid(db, fname, backend):
    obs = db.query("AgentEntry")
    simid = obs[0][0]
    for row in obs[1:]:
        assert_equal(simid, row[0])


@dbtest
def test_conds_ae(db, fname, backend):
    obs = db.query("AgentEntry", [('Kind', '==', 'Region')])
    assert_equal(1, len(obs))
    assert_equal('Region', obs[0][2])
    assert_equal(':agents:NullRegion', obs[0][3])


@dbtest
def test_conds_comp(db, fname, backend):
    conds = [('NucId', '==', 922350000), ('MassFrac', '>', 0.0072)]
    obs = db.query("Compositions", conds)
    assert_less(0, len(obs))
    for row in obs:
        assert_less(0.0072, row[-1])


if __name__ == "__main__":
    nose.runmodule()