"""Tests for cyclus wrappers"""
import os
import subprocess

import nose
from nose.tools import assert_equal, assert_less

from cymetric import cyclus

def setup():
    if os.path.isfile('test.h5'):
        return
    subprocess.check_call(['cyclus', '-otest.h5', 'test-input.xml'])

def test_hdf5_name():
    exp = "test.h5"
    db = cyclus.Hdf5Back(exp)
    obs = db.name()
    assert_equal(exp, obs)


def test_hdf5_simid():
    db = cyclus.Hdf5Back("test.h5")
    obs = db.query("AgentEntry")
    simid = obs[0][0]
    for row in obs[1:]:
        assert_equal(simid, row[0])


def test_hdf5_conds_ae():
    db = cyclus.Hdf5Back("test.h5")
    obs = db.query("AgentEntry", [('Kind', '==', 'Region')])
    assert_equal(1, len(obs))
    assert_equal('Region', obs[0][2])
    assert_equal(':agents:NullRegion', obs[0][3])


def test_hdf5_conds_comp():
    db = cyclus.Hdf5Back("test.h5")
    conds = [('NucId', '==', 922350000), ('MassFrac', '>', 0.0072)]
    obs = db.query("Compositions", conds)
    assert_less(0, len(obs))
    for row in obs:
        assert_less(0.0072, row[-1])


if __name__ == "__main__":
    nose.runmodule()