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

def test_annual_costs():
	"""
	"""
	# Reactor level
	assert_equal(cash_flows.annual_costs('test-output.sqlite', 20).sum().sum(),
				 cash_flows.annual_costs('test-output.sqlite', 21).sum().sum())
	assert_equal(cash_flows.annual_costs_present_value('test-output.sqlite', 20).sum().sum(),
				 cash_flows.annual_costs_present_value('test-output.sqlite', 21).sum().sum())
	# Region / Institution level
	assert_equal(cash_flows.region_annual_costs('test-output.sqlite', 11).sum().sum(),
				cash_flows.institution_annual_costs('test-output.sqlite', 12).sum().sum())
	assert_equal(cash_flows.region_annual_costs_present_value(
	'test-output.sqlite', 11).sum().sum(),
				 cash_flows.institution_annual_costs_present_value(
	'test-output.sqlite', 12).sum().sum())
	
def test_average_cost():
	"""
	"""
	assert_equal(cash_flows.average_cost('test-output.sqlite', 20),
				 cash_flows.average_cost('test-output.sqlite', 21))

def test_lcoe():
	"""
	"""
	assert_equal(cash_flows.lcoe('test-output.sqlite', 20),
				 cash_flows.lcoe('test-output.sqlite', 21))

def test_period_costs():
	"""
	"""
	# Reactor level
	assert_equal(cash_flows.period_costs('test-output.sqlite', 20, 30).sum().sum(),
				 cash_flows.period_costs('test-output.sqlite', 21, 30).sum().sum())
	# Region / Institution level
	assert_equal(cash_flows.region_period_costs('test-output.sqlite', 11, 25).sum().sum(),
				cash_flows.institution_period_costs('test-output.sqlite', 12, 25).sum().sum())
				
"""
def test_iter_calculation():
    Important question : testing all metrics ? See if it runs and if we can see small changes (gaussian distribution), calculations should be verified in test_eco_metrics
    inspired by test_eval
    
    df = cash_flows.iter_calculations(10, test-output.sqlite, 'CapitalCost')
    assert_less(0, len(df))
"""