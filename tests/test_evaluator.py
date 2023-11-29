"""Tests for evaluator"""
from __future__ import print_function, unicode_literals

from cymetric import evaluator

from tools import dbtest


def test_eval(dbtest):
    db, fname, backend = dbtest
    df = evaluator.eval('Materials', db, write=False)
    assert 0 < len(df)

