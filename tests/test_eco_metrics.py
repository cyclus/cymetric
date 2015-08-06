"""Tests for metrics. These test metric calculation functions unbound to any 
database. This makes writing the tests easier in a unit test like fashion.
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
    """Plan for the test : first, with own param (one default) : same as usual, see if obs=exp
    This test shows that the capital_cost function can calculate the cost given the parameters, and that the different levels of parameters can be read : one at the facility level, the other at region level...
    To verify that random parameters work, only need to test the iter function and see if we obtain different values
    """
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 0, 112312.244694),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 1, 114282.634951),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 2, 116253.025209),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 3, 118223.415467),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 4, 115760.427645)
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


if __name__ == "__main__":
    nose.runmodule()