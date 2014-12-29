"""Tests for evaluator"""
from __future__ import print_function, unicode_literals

import nose
from nose.tools import assert_equal, assert_less

from cymetric import evaluator
from cymetric import metrics
from cymetric import root_metrics

from tools import setup, dbtest

@dbtest
def test_eval(db, fname, backend):
    df = evaluator.eval('Materials', db)
    assert_less(0, len(df))
    #assert False

if __name__ == "__main__":
    nose.runmodule()