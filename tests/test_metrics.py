"""Tests for metrics. These test metric calculation functions
unbound to any database. This makes writing the tests easier in a unit
test like fashion.
"""
from __future__ import print_function, unicode_literals
from uuid import UUID

import nose
from nose.tools import assert_equal, assert_less

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from cymetric import cyclus
from cymetric import metrics
from cymetric.tools import raw_to_series, ensure_dt_bytes


def test_agents():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 22, 'Region', ':agents:NullRegion', 'USA', -1, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 23, 'Inst', ':agents:NullInst', 'utility', 22, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 24, 'Facility', ':agents:Source', 'MineU235', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 25, 'Facility', ':agents:Source', 'U238', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 26, 'Facility', ':agents:Source', 'DU', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 'Facility', ':agents:Source', 'DU2', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 28, 'Facility', ':Brightlite:FuelfabFacility', 'LWR Fuel FAb', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 29, 'Facility', ':Brightlite:ReprocessFacility', 'LWR Seperation', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 30, 'Facility', ':Brightlite:ReprocessFacility', 'FR Reprocess', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 31, 'Facility', ':agents:Sink', 'SINK', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 32, 'Facility', ':Brightlite:FuelfabFacility', 'FR Fuel Fab', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 33, 'Inst', ':cycamore:DeployInst', 'utility2', 22, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 34, 'Facility', ':Brightlite:ReactorFacility', 'LWR', 33, -1, 5, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 35, 'Facility', ':Brightlite:ReactorFacility', 'LWR', 33, -1, 5, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 36, 'Facility', ':Brightlite:ReactorFacility', 'FRx', 33, -1, 10, 120.0),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('AgentId', '<i8'), 
                ('Kind', 'O'), ('Spec', 'O'), ('Prototype', 'O'), 
                ('ParentId', '<i8'), ('Lifetime', '<i8'), 
                ('EnterTime', '<i8'), ('ExitTime', '<f8')]))
        )
    agent_entry = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 22, 'Region', ':agents:NullRegion', 'USA', -1, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 23, 'Inst', ':agents:NullInst', 'utility', 22, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 24, 'Facility', ':agents:Source', 'MineU235', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 25, 'Facility', ':agents:Source', 'U238', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 26, 'Facility', ':agents:Source', 'DU', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 'Facility', ':agents:Source', 'DU2', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 28, 'Facility', ':Brightlite:FuelfabFacility', 'LWR Fuel FAb', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 29, 'Facility', ':Brightlite:ReprocessFacility', 'LWR Seperation', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 30, 'Facility', ':Brightlite:ReprocessFacility', 'FR Reprocess', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 31, 'Facility', ':agents:Sink', 'SINK', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 32, 'Facility', ':Brightlite:FuelfabFacility', 'FR Fuel Fab', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 33, 'Inst', ':cycamore:DeployInst', 'utility2', 22, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 34, 'Facility', ':Brightlite:ReactorFacility', 'LWR', 33, -1, 5),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 35, 'Facility', ':Brightlite:ReactorFacility', 'LWR', 33, -1, 5),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 36, 'Facility', ':Brightlite:ReactorFacility', 'FRx', 33, -1, 10),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('AgentId', '<i8'), 
                ('Kind', 'O'), ('Spec', 'O'), ('Prototype', 'O'), 
                ('ParentId', '<i8'), ('Lifetime', '<i8'), 
                ('EnterTime', '<i8')]))
        )
    info = pd.DataFrame({'Duration': {0: 120}, 
        'SimId': {0: UUID('f22f2281-2464-420a-8325-37320fd418f8')}, 
        })
    dur = info.set_index(['SimId'])
    series = [raw_to_series(agent_entry, ['SimId', 'AgentId'], col) \
              for col in ('Kind', 'Spec', 'Prototype', 'ParentId', 
                          'Lifetime', 'EnterTime')]
    series += [None, None, dur]
    obs = metrics.agents.func(series)
    assert_frame_equal(exp, obs)


def test_materials():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922350000, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922380000, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 942390000, 0),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('QualId', '<i8'), ('ResourceId', '<i8'), ('ObjId', '<i8'), 
                ('TimeCreated', '<i8'), ('NucId', '<i8'), ('Mass', '<f8')]))
        )
    resources = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 9, 7, 'Material', 1, 1, 'kg', 5, 0, 0),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('ResourceId', '<i8'), ('ObjId', '<i8'), 
                ('Type', 'O'), ('TimeCreated', '<i8'), ('Quantity', '<i8'), 
                ('Units', 'O'), ('QualId', '<i8'), ('Parent1', '<i8'), 
                ('Parent2', '<i8')]))
        )
    compositions = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 922350000, 0.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 922380000, 0.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 942390000, 0.0),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('QualId', '<i8'), ('NucId', '<i8'), 
                ('MassFrac', '<f8')]))
        )
#    info = pd.DataFrame({'Mass': {0: 120}, 
#        'SimId': {0: UUID('f22f2281-2464-420a-8325-37320fd418f8')}, 
#        })
#    mass = info.set_index(['SimId'])
    series = [raw_to_series(materials, ['SimId', 'QualId'], col) \
              for col in ('ResourceId', 'ObjId', 'TimeCreated', 'NucId', 'Mass')]
#    series += [None, None, mass]
    obs = metrics.materials.func(series)
    assert_frame_equal(exp, obs)


if __name__ == "__main__":
    nose.runmodule()
