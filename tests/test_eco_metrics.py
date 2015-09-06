"""Tests for economic metrics and derived functions.
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
from cymetric.tools import raw_to_series, ensure_dt_bytes


def test_capital_cost():
    """
    """
    exp = pd.DataFrame(np.array([
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 0, 0.0945),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 1, 0.1050),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 2, 112312.244694),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 3, 112312.244694),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 4, 112312.244694),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 5, 112312.244694),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 6, 112312.244694),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 7, 112312.244694),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 8, 112312.244694),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 9, 112312.244694),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 10, 112312.244694),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 11, 112312.244694)
        ], dtype=ensure_dt_bytes([
        		('SimId','O'), ('AgentId', '<i8'), ('Time','<i8'),
        		('Payment', '<f8')]))
        )
    power = pd.DataFrame(np.array([
          (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 1, 3),
          (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 1, 4)
          ], dtype=ensure_dt_bytes([
                  ('SimId', 'O'), ('AgentId', '<i8'), ('Value', '<f8'),
                  ('Time', '<i8')]))
          )
    entry = pd.DataFrame(np.array([
          (13, 1, ':cycamore:Reactor', 3),
          (1, -1, ':cycamore:DeployInst', 2)
          ], dtype=ensure_dt_bytes([('AgentId', '<i8'), ('ParentId', '<i8'),
          		('Spec', 'O'), ('EnterTime', '<i8')]))
          )
    info = pd.DataFrame(np.array([
    	 (2005, 4, 5)
    	 ], dtype=ensure_dt_bytes([('InitialYear', '<i8'),
    	 		 ('InitialMonth', '<i8'), ('Duration', '<i8')]))
    	 )
    s1 = power.set_index(['SimId', 'AgentId', 'Value'])['Time']
    s2 = entry.set_index(['AgentId', 'ParentId', 'Spec'])['EnterTime']
    s3 = info.set_index(['InitialYear', 'InitialMonth'])['Duration']
    series = [s1, s2, s3]
    obs = eco_metrics.capital_cost.func(series)
    assert_frame_equal(exp, obs)


def test_fuel_cost():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 34, 1, 'uox', 
        29641.600000000002, 46),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 11, 3, 'mox', 12980.0, 9)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('TransactionId', '<i8'), ('AgentId','<i8'),
             ('Commodity', 'O'), ('Payment', '<f8'), ('Time', '<i8')]))
        )
    resources = pd.DataFrame(np.array([
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 12.56),
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 5.5),
              ], dtype=ensure_dt_bytes([
                      ('SimId', 'O'), ('ResourceId', '<i8'), ('Quantity', 
                      '<f8'),]))
              )
    transactions = pd.DataFrame(np.array([
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 34, 1, 27,
                 'uox', 46),
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 11, 3, 13, 
                 'mox', 9)
                 ], dtype=ensure_dt_bytes([
                         ('SimId', 'O'), ('TransactionId', '<i8'),
                         ('ReceiverId', '<i8'), ('ResourceId', '<i8'), 
                         ('Commodity', 'O'), ('Time', '<i8')]))
                 )
    s1 = resources.set_index(['SimId', 'ResourceId'])['Quantity']
    s2 = transactions.set_index(['SimId', 'TransactionId', 'ReceiverId', 
       'ResourceId', 'Commodity'])['Time']
    series = [s1, s2]
    obs = eco_metrics.fuel_cost.func(series)
    assert_frame_equal(exp, obs)


def test_decommissioning_cost():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 0, 60),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 1516.778523, 61),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 3033.557047, 62),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 4550.335570, 63),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 6067.114094, 64)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('AgentId', '<i8'), ('Payment','<f8'),
             ('Time', '<i8')]))
        )
    power = pd.DataFrame(np.array([
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 11.3)
                 ], dtype=ensure_dt_bytes([
                         ('SimId', 'O'), ('AgentId', '<i8'),
                         ('Value', '<f8')]))
                 )
    entry = pd.DataFrame(np.array([
                 (10, 50, 27, ':cycamore:Reactor'),
                 (1, 40, 5, ':cycamore:Sink')
                 ], dtype=ensure_dt_bytes([
                         ('EnterTime', '<i8'), ('Lifetime', '<i8'),
                         ('AgentId', '<i8'), ('Spec', 'O')]))
                 )
    info = pd.DataFrame(np.array([
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 2011, 8, 65)
              ], dtype=ensure_dt_bytes([
                      ('SimId', 'O'), ('InitialYear', '<i8'),
                      ('InitialMonth', '<i8'),('Duration', '<i8')]))
              )
    s1 = power.set_index(['SimId', 'AgentId'])['Value']
    s2 = entry.set_index(['EnterTime', 'Lifetime', 'AgentId'])['Spec']
    s3 = info.set_index(['SimId', 'InitialYear', 'InitialMonth'])['Duration']
    series = [s1, s2, s3]
    obs = eco_metrics.decommissioning_cost.func(series)
    assert_frame_equal(exp, obs)


def test_operation_maintenance():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 0, 25436.85),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 2, 25436.85),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 8, 43800.0)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('AgentId', '<i8'), ('Time', '<i8'),
             ('Payment','<f8')]))
        )
    power = pd.DataFrame(np.array([
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 0, 2.323),
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 2, 2.323),
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 8, 4.0)
              ], dtype=ensure_dt_bytes([
                      ('SimId', 'O'), ('AgentId', '<i8'), ('Time', '<i8'),
                      ('Value', '<f8')]))
              )
    s1 = power.set_index(['SimId', 'AgentId', 'Time'])['Value']
    series = [s1]
    obs = eco_metrics.operation_maintenance.func(series)
    assert_frame_equal(exp, obs)
    
def test_economic_info():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 0, 25436.85),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 2, 25436.85),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 8, 43800.0)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('AgentId', '<i8'), ('Time', '<i8'),
             ('Payment','<f8')]))
        )
    power = pd.DataFrame(np.array([
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 0, 2.323),
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 2, 2.323),
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 8, 4.0)
              ], dtype=ensure_dt_bytes([
                      ('SimId', 'O'), ('AgentId', '<i8'), ('Time', '<i8'),
                      ('Value', '<f8')]))
              )
    s1 = power.set_index(['SimId', 'AgentId', 'Time'])['Value']
    series = [s1]
    obs = eco_metrics.operation_maintenance.func(series)
    assert_frame_equal(exp, obs)


##### Tests of the functions derived from the 5 basic economic metrics #####

"""test_eco_metrics is supposed to test all side features whereas test_eco_metrics test the calculations of the metrics in a more conventional way (calculating one obs and comparing to an exp)
"""

def test_annual_costs():
	"""
	"""
	# Reactor / Institution level
	assert_equal(eco_metrics.annual_costs('test_output.sqlite', 13).sum().sum(),
				 eco_metrics.institution_annual_costs('test_output.sqlite', 9).sum().sum())
	assert_equal(eco_metrics.annual_costs_present_value('test_output.sqlite', 13).sum().sum(),
				 eco_metrics.institution_annual_costs_present_value(
	'test_output.sqlite', 9).sum().sum())
	# Region / Institution level
	assert_equal(eco_metrics.region_annual_costs('test_output.sqlite', 8).sum().sum(),
				eco_metrics.institution_annual_costs('test_output.sqlite', 9).sum().sum())
	assert_equal(eco_metrics.region_annual_costs_present_value(
	'test_output.sqlite', 8).sum().sum(),
				 eco_metrics.institution_annual_costs_present_value(
	'test_output.sqlite', 9).sum().sum())	
	# Simulation / Reactor level
	assert_equal(eco_metrics.region_annual_costs('test_output.sqlite', 8).sum().sum(),
				eco_metrics.simulation_annual_costs('test_output.sqlite').sum().sum())
	assert_equal(eco_metrics.region_annual_costs_present_value(
	'test_output.sqlite', 8).sum().sum(),
				 eco_metrics.simulation_annual_costs_present_value(
	'test_output.sqlite').sum().sum())

def test_lcoe():
	"""
	"""
	# Reactor / Institution level
	assert_equal(eco_metrics.lcoe('test_output.sqlite', 13),
				 eco_metrics.institution_lcoe('test_output.sqlite', 9))
	# Region / Institution level
	assert_equal(eco_metrics.region_lcoe('test_output.sqlite', 8),
				 eco_metrics.institution_lcoe('test_output.sqlite', 9))
	# Simulation / Reactor level
	assert_equal(eco_metrics.region_lcoe('test_output.sqlite', 8),
				 eco_metrics.simulation_lcoe('test_output.sqlite'))
				 
def test_average_lcoe():
	"""
	"""
	# Reactor / Institution level
	assert_equal(eco_metrics.average_lcoe('test_output.sqlite', 13),
				 eco_metrics.institution_average_lcoe('test_output.sqlite', 9))
	# Region / Institution level
	assert_equal(eco_metrics.region_average_lcoe('test_output.sqlite', 8),
				 eco_metrics.institution_average_lcoe('test_output.sqlite', 9))
	# Simulation / Reactor level
	assert_equal(eco_metrics.region_average_lcoe('test_output.sqlite', 8),
				 eco_metrics.simulation_average_lcoe('test_output.sqlite'))
				 
def test_benefit():
	"""
	"""
	# Reactor / Institution level
	assert_equal(eco_metrics.benefit('test_output.sqlite', 13),
				 eco_metrics.institution_benefit('test_output.sqlite', 9))
	# Region / Institution level
	assert_equal(eco_metrics.region_benefit('test_output.sqlite', 8),
				 eco_metrics.institution_benefit('test_output.sqlite', 9))
	# Simulation / Reactor level
	assert_equal(eco_metrics.region_benefit('test_output.sqlite', 8),
				 eco_metrics.simulation_benefit('test_output.sqlite'))
				 
def test_power_generated():
	"""
	"""
	# Reactor / Institution level
	assert_equal(eco_metrics.power_generated('test_output.sqlite', 13),
				 eco_metrics.institution_power_generated('test_output.sqlite', 9))
	# Region / Institution level
	assert_equal(eco_metrics.region_power_generated('test_output.sqlite', 8),
				 eco_metrics.institution_power_generated('test_output.sqlite', 9))
	# Simulation / Reactor level
	assert_equal(eco_metrics.region_power_generated('test_output.sqlite', 8),
				 eco_metrics.simulation_power_generated('test_output.sqlite'))

def test_period_costs():
	"""
	"""
	# Reactor level
	assert_equal(eco_metrics.period_costs('test_output.sqlite', 13, 30).sum().sum(),
				 eco_metrics.period_costs2('test_output.sqlite', 13, 30).sum().sum())
	# Reactor / Institution level
	assert_equal(eco_metrics.period_costs('test_output.sqlite', 13).sum().sum(),
				eco_metrics.institution_period_costs('test_output.sqlite', 9).sum().sum())
	# Region / Institution level
	assert_equal(eco_metrics.region_period_costs('test_output.sqlite', 8).sum().sum(),
				eco_metrics.institution_period_costs('test_output.sqlite', 9).sum().sum())
	# Region / Simulation level
	assert_equal(eco_metrics.region_period_costs('test_output.sqlite', 8).sum().sum(),
				eco_metrics.simulation_period_costs('test_output.sqlite').sum().sum())
				

if __name__ == "__main__":
    nose.runmodule()