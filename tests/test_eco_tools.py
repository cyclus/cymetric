"""Tests for functions in eco_tools.py"""
from __future__ import print_function, unicode_literals

import nose
from nose.tools import assert_equal, assert_less

import pandas as pd

from cymetric import evaluator
from cymetric import metrics
from cymetric import root_metrics
from tools import setup, dbtest


@dbtest
def test_raw_to_series(db, fname, backend):
    df = evaluator.eval('Materials', db)
    df2 = evaluator.eval('Resources', db)
    assert_equal(isinstance(df2['Quantity'].index, pd.MultiIndex), False)

if __name__=="__main__":
    nose.runmodule()