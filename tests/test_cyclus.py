"""Tests for cyclus wrappers"""
import os
import subprocess

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
