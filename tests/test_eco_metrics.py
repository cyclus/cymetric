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

from cymetric import eco_metrics
from cymetric.tools import raw_to_series, ensure_dt_bytes


def test_capital_cost():
    """
    """
    exp = pd.DataFrame(np.array([
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 0, 0.126000),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 1, 0.139965),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 2, 0.112035),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 3, 0.084000),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 4, 0.055965),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 5, 0.028035),
        (UUID('4375a413-dafb-4da1-bfcc-f16e59b5a3e0'), 13, 6, 0.0)
        ], dtype=ensure_dt_bytes([
                ('SimId','O'), ('AgentId', '<i8'), ('Time','<i8'),
                ('Payment', '<f8')]))
        )
    power = pd.DataFrame(np.array([
          (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 1, 1)
          ], dtype=ensure_dt_bytes([
                  ('SimId', 'O'), ('AgentId', '<i8'), ('Value', '<f8'),
                  ('Time', '<i8')]))
          )
    entry = pd.DataFrame(np.array([
          (13, -1, ':cycamore:Reactor', 1)
          ], dtype=ensure_dt_bytes([('AgentId', '<i8'), ('ParentId', '<i8'),
                  ('Spec', 'O'), ('EnterTime', '<i8')]))
          )
    info = pd.DataFrame(np.array([
         (2000, 1, 20)
         ], dtype=ensure_dt_bytes([('InitialYear', '<i8'),
                  ('InitialMonth', '<i8'), ('Duration', '<i8')]))
         )
    ecoInfo = pd.DataFrame(np.array([
         ('Reactor1', 13, 10, 5, 10, 0, 1, 0.1)
         ], dtype=ensure_dt_bytes([('Prototype', 'O'), ('AgentId',
             '<i8'), ('Capital_beforePeak', '<i8'),
             ('Capital_afterPeak', '<i8'), ('Capital_constructionDuration', '<i8'),
             ('Capital_Deviation', '<f8'), ('Capital_OvernightCost', '<f8'),
             ('Finance_DiscountRate', '<f8')]))
         )
    obs = eco_metrics.capital_cost.func(power, entry, info, ecoInfo)
    assert_frame_equal(exp.drop(['SimId'], axis=1), obs.drop(['SimId'], axis=1))


def test_fuel_cost():
    exp = pd.DataFrame(np.array([
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 0, 13, 'uox', 1, 1),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 1, 13, 'uox', 1, 2),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 2, 13, 'uox', 1, 3),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 3, 13, 'uox', 1, 4),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 4, 13, 'uox', 1, 5),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 5, 13, 'uox', 1, 6),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 6, 13, 'uox', 1, 7),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 7, 13, 'uox', 1, 8),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 8, 13, 'uox', 1, 9)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('TransactionId', '<i8'), ('AgentId','<i8'),
             ('Commodity', 'O'), ('Payment', '<f8'), ('Time', '<i8')]))
        )
    resources = pd.DataFrame(np.array([
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 5, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 6, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 11, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 12, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 17, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 18, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 23, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 24, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 29, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 30, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 35, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 36, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 41, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 42, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 47, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 48, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 53, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 54, 1)
              ], dtype=ensure_dt_bytes([
                      ('SimId', 'O'), ('ResourceId', '<i8'), ('Quantity',
                      '<f8'),]))
              )
    transactions = pd.DataFrame(np.array([
                 (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 0, 12, 13, 5,
                 'uox', 1),
                 (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 1, 12, 13, 11,
                 'uox', 2),
                 (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 2, 12, 13, 17,
                 'uox', 3),
                 (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 3, 12, 13, 23,
                 'uox', 4),
                 (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 4, 12, 13, 29,
                 'uox', 5),
                 (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 5, 12, 13, 35,
                 'uox', 6),
                 (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 6, 12, 13, 41,
                 'uox', 7),
                 (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 7, 12, 13, 47,
                 'uox', 8),
                 (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 8, 12, 13, 53,
                 'uox', 9)
                 ], dtype=ensure_dt_bytes([
                         ('SimId', 'O'), ('TransactionId', '<i8'),
                         ('ReceiverId', '<i8'), ('SenderId', '<i8'), ('ResourceId', '<i8'),
                         ('Commodity', 'O'), ('Time', '<i8')]))
                 )
    ecoInfo = pd.DataFrame(np.array([
                 ('Reactor1', 13, 'uox', 1, 0, 0, 0.1)
                 ], dtype=ensure_dt_bytes([
                         ('Agent_Prototype', 'O'), ('Agent_AgentId', '<i8'),
                         ('Fuel_Commodity', 'O'), ('Fuel_SupplyCost', '<f8'),
                         ('Fuel_WasteFee', '<f8'), ('Fuel_Deviation', '<f8'),
                         ('Finance_DiscountRate', '<f8')]))
                 )
    s1 = resources.set_index(['SimId', 'ResourceId'])['Quantity']
    s2 = transactions.set_index(['SimId', 'TransactionId', 'ReceiverId',
       'ResourceId', 'Commodity'])['Time']
    s3 = ecoInfo.set_index(['Agent_Prototype', 'Agent_AgentId',
        'Fuel_Commodity', 'Fuel_SupplyCost', 'Fuel_WasteFee',
        'Fuel_Deviation', 'Finance_DiscountRate'])
    series = [s1, s2, s3]
    obs = eco_metrics.fuel_cost.func(series)
    assert_frame_equal(exp, obs)


def test_decommissioning_cost():
    exp = pd.DataFrame(np.array([
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 0.0, 11),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 0.133333, 12),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 0.266667, 13),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 0.4, 14),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 0.2, 15),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 0.0, 16)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('AgentId', '<i8'), ('Payment','<f8'),
             ('Time', '<i8')]))
        )
    power = pd.DataFrame(np.array([
                 (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 1)
                 ], dtype=ensure_dt_bytes([
                         ('SimId', 'O'), ('AgentId', '<i8'),
                         ('Value', '<f8')]))
                 )
    entry = pd.DataFrame(np.array([
                 (1, 10, 13, ':cycamore:Reactor')
                 ], dtype=ensure_dt_bytes([
                         ('EnterTime', '<i8'), ('Lifetime', '<i8'),
                         ('AgentId', '<i8'), ('Spec', 'O')]))
                 )
    info = pd.DataFrame(np.array([
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 2000, 1, 20)
              ], dtype=ensure_dt_bytes([
                      ('SimId', 'O'), ('InitialYear', '<i8'),
                      ('InitialMonth', '<i8'),('Duration', '<i8')]))
              )
    ecoInfo = pd.DataFrame(np.array([
              (13, 5, 1)
              ], dtype=ensure_dt_bytes([
                      ('Agent_AgentId', '<i8'), ('Decommissioning_Duration',
                          '<f8'), ('Decommissioning_OvernightCost', '<f8')]))
              )
    s1 = power.set_index(['SimId', 'AgentId'])['Value']
    s2 = entry.set_index(['EnterTime', 'Lifetime', 'AgentId'])['Spec']
    s3 = info.set_index(['SimId', 'InitialYear', 'InitialMonth'])['Duration']
    s4 = ecoInfo.set_index(['Agent_AgentId', 'Decommissioning_Duration'])['Decommissioning_OvernightCost']
    series = [s1, s2, s3, s4]
    obs = eco_metrics.decommissioning_cost.func(series)
    assert_frame_equal(exp, obs)


def test_operation_maintenance():
    exp = pd.DataFrame(np.array([
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 1, 731),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 2, 731),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 3, 731),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 4, 731),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 5, 731),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 6, 731),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 7, 731),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 8, 731),
        (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 9, 731)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('AgentId', '<i8'), ('Time', '<i8'),
             ('Payment','<f8')]))
        )
    power = pd.DataFrame(np.array([
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 1, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 2, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 3, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 4, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 5, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 6, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 7, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 8, 1),
              (UUID('0ac0f445-3e1c-43ec-826c-8702d4fc2f40'), 13, 9, 1)
              ], dtype=ensure_dt_bytes([
                      ('SimId', 'O'), ('AgentId', '<i8'), ('Time', '<i8'),
                      ('Value', '<f8')]))
              )
    ecoInfo = pd.DataFrame(np.array([
              (13, 1, 1, 0)
              ], dtype=ensure_dt_bytes([
                      ('AgentId', '<i8'),
                      ('FixedCost', '<f8'),
                      ('VariableCost', '<f8'),
                      ('Deviation', '<f8')]))
              )
    #s1 = power.set_index(['SimId', 'AgentId', 'Time'])['Value']
    #s2 = ecoInfo.set_index(['AgentId', 'FixedCost', 'VariableCost'])[ 'Deviation']
    obs = eco_metrics.operation_maintenance.func(power, ecoInfo)
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
    entry = pd.DataFrame(np.array([
              ('Region1', 8, -1, 0.1, 0.1, 0.1, 0.1, 10, 5, 10, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20),
              ('Institution1', 9, 8, 0.1, 0.1, 0.1, 0.1, 10, 5, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20),
              ('Source', 12, 9, 0.1, 0.1, 0.1, 0.1, 10, 5, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20),
              ('Reactor1', 13, 9, 0.1, 0.1, 0.1, 0.1, 10, 5, 10, 1, 1, 1, 0, 0,
                  0, 0, 'uox', 1, 0, 0, 0, 20)
              ], dtype=ensure_dt_bytes([
                      ('Agent_Prototype', 'O'), ('Agent_AgentId', '<i8'),
                      ('Agent_ParentId', '<i8'), ('Finance_ReturnOnDebt',
                          '<f8'), ('Finance_ReturnOnEquity', '<f8'),
                      ('Finance_TaxRate', '<f8'), ('Finance_DiscountRate',
                          '<f8'), ('Captial_beforePeak', '<i8'),
                      ('Captial_afterPeak', '<i8'),
                      ('Captial_constructionDuration', '<i8'),
                      ('Captial_Deviation', '<f8'), ('Captial_OvernightCost',
                          '<f8'), ('Decommissioning_Duration', '<i8'),
                      ('Decommissioning_OvernightCost', '<f8'),
                      ('OperationMaintenance_FixedCost', '<f8'),
                      ('OperationMaintenance_VariableCost', '<f8'),
                      ('OperationMaintenance_Deviation', '<f8'),
                      ('Fuel_Commodity', 'O'), ('Fuel_SupplyCost', '<f8'),
                      ('Fuel_WasteFee', '<f8'), ('Fuel_Deviation', '<f8'),
                      ('Truncation_Begin', '<i8'), ('Truncation_End', '<i8')]))
              )
    s1 = entry.set_index(['Agent_AgentId', 'Agent_Prototype'])['Agent_ParentId']
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
    assert_equal(eco_metrics.lcoe('test_output.sqlite', 13),
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
