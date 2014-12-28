"""Tests for root metrics"""
from __future__ import print_function, unicode_literals
import os
import subprocess
from functools import wraps

import nose
from nose.tools import assert_equal, assert_less

from cymetric import cyclus
from cymetric import root_metrics

from tools import setup, dbtest

@dbtest
def test_resources(db, fname, backend):
    r = root_metrics.resources(db=db)
    obs = r()
    assert_less(0, len(obs))
    assert_equal('Resources', r.__name__)


if __name__ == "__main__":
    nose.runmodule()