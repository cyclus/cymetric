"""Tests for cyclus wrappers"""
from nose.tools import assert_equal

from cymetric import cyclus

def test_hdf5_name():
    exp = "test.h5"
    db = cyclus.Hdf5Back(exp)
    obs = db.name()
    assert_equal(exp, obs)
