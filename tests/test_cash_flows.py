"""test_cash_flows is supposed to test all side features whereas test_eco_metrics test the calculations of the metrics in a more conventional way (calculating one obs and comparing to an exp)
"""
from __future__ import print_function, unicode_literals
from uuid import UUID

import nose
from nose.tools import assert_equal, assert_less
from nose.plugins.skip import SkipTest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from cymetric import cyclus
from cymetric import eco_metrics
from cymetric import cash_flows
from cymetric.tools import raw_to_series, ensure_dt_bytes


				
"""
def test_iter_calculation():
    Important question : testing all metrics ? See if it runs and if we can see small changes (gaussian distribution), calculations should be verified in test_eco_metrics
    inspired by test_eval
    
    df = cash_flows.iter_calculations(10, test-output.sqlite, 'CapitalCost')
    assert_less(0, len(df))
"""