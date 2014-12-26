"""Tests for cyclus wrappers"""
import os
import subprocess

import nose
from nose.tools import assert_equal

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


def test_hdf5_name():
    db = cyclus.Hdf5Back("test.h5")
    obs = db.query("AgentEntry")
    #assert_equal(exp, obs)
    print(obs)
    assert False

if __name__ == "__main__":
    nose.runmodule()