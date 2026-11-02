"""Tests for dataframe helpers."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cymetric.tools import merge_and_fillna_col


@pytest.mark.parametrize('how', ['inner', 'left', 'right', 'outer'])
@pytest.mark.parametrize('index', [None, [10, 20, 30], [10, 10, 20]])
def test_merge_and_fillna_col_preserves_left_rows(how, index):
    left = pd.DataFrame({
        'AgentId': [2, 1, 3],
        'ExitTime': [np.nan, np.nan, 30.0],
    }, index=index)
    # Right-only rows must not replace a left row or introduce extra agents.
    right = pd.DataFrame({
        'AgentId': [1, 2, 4, 5],
        'Exits': [10.0, 20.0, 40.0, 50.0],
    })
    original = left.copy(deep=True)
    expected = left.copy()
    expected['ExitTime'] = [20.0, 10.0, 30.0]

    observed = merge_and_fillna_col(
        left, right, 'ExitTime', 'Exits', how=how, on=['AgentId'])

    assert_frame_equal(expected, observed)
    assert_frame_equal(original, left)


def test_merge_and_fillna_col_repeated_left_keys():
    left = pd.DataFrame({
        'SimId': [2, 1, 2],
        'ExitTime': [np.nan, 10.0, 25.0],
    }, index=[5, 7, 9])
    right = pd.DataFrame({'SimId': [1, 2], 'Duration': [100.0, 200.0]})
    expected = left.copy()
    expected['ExitTime'] = [200.0, 10.0, 25.0]

    observed = merge_and_fillna_col(
        left, right, 'ExitTime', 'Duration', how='outer', on=['SimId'])

    assert_frame_equal(expected, observed)
