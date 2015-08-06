"""Tests for functions in tools.py"""

import nose
from nose.tools import assert_equal

import pandas as pd

from cymetric import evaluator
from tools import dbtest


@dbtest
def test_raw_to_series(db, fname, backend):
    df = evaluator.eval('Materials', db)
    df2 = evaluator.eval('Resources', db)
    assert_equal(isinstance(df2['Quantity'].index, pd.MultiIndex), False)
