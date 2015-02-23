"""A collection of metrics that come stock with cymetric.
"""
from __future__ import print_function, unicode_literals
import inspect

import numpy as np
import pandas as pd

from cymetric import cyclus
from cymetric import schemas
from cymetric import typesystem as ts
from cymetric import tools
from cymetric.evaluator import register_metric

from pyne import data


class Metric(object):

    dependencies = NotImplemented
    schema = NotImplemented
    
    def __init__(self, db):
        self.db = db

    @property
    def name(self):
        return self.__class__.__name__


def _genmetricclass(f, name, depends, scheme):
    if not isinstance(scheme, schemas.schema):
        scheme = schemas.schema(scheme)

    class Cls(Metric):
        dependencies = depends
        schema = scheme
        func = staticmethod(f)

        __doc__ = inspect.getdoc(f)

        def __init__(self, db):
            super(Cls, self).__init__(db)

        def __call__(self, series, conds=None, known_tables=None, *args, **kwargs):
            # FIXME test if I already exist in the db, read in if I do
            if known_tables is None:
                known_tables = self.db.tables()
            if self.name in known_tables:
                return self.db.query(self.name, conds=conds)
            return f(series)

    Cls.__name__ = str(name)
    register_metric(Cls)
    return Cls


def metric(name=None, depends=NotImplemented, schema=NotImplemented):
    def dec(f):
        clsname = name or f.__name__
        return _genmetricclass(f=f, name=clsname, scheme=schema, depends=depends)
    return dec


#
# The actual metrics
#

# Material Mass (quantity * massfrac)
_matdeps = [('Resources', ('SimId', 'QualId', 'ResourceId', 'ObjId', 'TimeCreated'), 
                'Quantity'),
            ('Compositions', ('SimId', 'QualId', 'NucId'), 'MassFrac')]

_matschema = (('SimId', ts.UUID), ('QualId', ts.INT), 
              ('ResourceId', ts.INT), ('ObjId', ts.INT), 
              ('TimeCreated', ts.INT), ('NucId', ts.INT), 
              ('Mass', ts.DOUBLE))

@metric(name='Materials', depends=_matdeps, schema=_matschema)
def materials(series):
    x = pd.merge(series[0].reset_index(), series[1].reset_index(), 
            on=['SimId', 'QualId'], how='inner').set_index(['SimId', 'QualId', 
                'ResourceId', 'ObjId','TimeCreated', 'NucId'])
    y = x['Quantity'] * x['MassFrac']
    y.name = 'Mass'
    z = y.reset_index()
    return z

del _matdeps, _matschema


# Activity (mass * decay_const / atomic_mass)
_actdeps = [('Materials', ('ResourceId',), 'Mass'), 
            ('Materials', ('ResourceId',), 'NucId')]

_actschema = [('ResourceId', ts.INT), ('Activity', ts.DOUBLE)]

@metric(name='Activity', depends=_actdeps, schema=_actschema)
def activity(series):
    mass = series[0]
    nucid = series[1]
    for nuc, m in zip(nucid, mass):
        act.append = (1000 * data.N_A * mass * data.decay_const(nucid) \
                     / data.atomic_mass(nucid))
    act.name = 'Activity'
# how to store in actschema use group by to group by resourceid then make act per nuc
    rtn = act.reset_index()
    return rtn

del _actdeps, _actschema


# Agents

_agentsdeps = [
    ('AgentEntry', ('SimId', 'AgentId'), 'Kind'),
    ('AgentEntry', ('SimId', 'AgentId'), 'Spec'),
    ('AgentEntry', ('SimId', 'AgentId'), 'Prototype'),
    ('AgentEntry', ('SimId', 'AgentId'), 'ParentId'),
    ('AgentEntry', ('SimId', 'AgentId'), 'Lifetime'),
    ('AgentEntry', ('SimId', 'AgentId'), 'EnterTime'),
    ('AgentExit', ('SimId', 'AgentId'), 'ExitTime'),
    ('DecomSchedule', ('SimId', 'AgentId'), 'DecomTime'),
    ('Info', ('SimId',), 'Duration'),
    ]

_agentsschema = schemas.schema([
    ('SimId', ts.UUID), ('AgentId', ts.INT), ('Kind', ts.STRING), 
    ('Spec', ts.STRING), ('Prototype', ts.STRING), ('ParentId', ts.INT), 
    ('Lifetime', ts.INT), ('EnterTime', ts.INT), ('ExitTime', ts.INT),
    ])

@metric(name='Agents', depends=_agentsdeps, schema=_agentsschema)
def agents(series):
    """Computes the Agents table. This is tricky because both the AgentExit
    table and the DecomSchedule table may not be present in the database.
    Furthermore, the Info table does not contain the AgentId column. This
    computation handles the calculation of the ExitTime in the face a 
    significant amounts of missing data.
    """
    mergeon  = ['SimId', 'AgentId']
    idx = series[0].index
    df = series[0].reset_index()
    for s in series[1:6]:
        df = pd.merge(df, s.reset_index(), on=mergeon)
    agent_exit = series[6]
    if agent_exit is None:
        agent_exit = pd.Series(index=idx, data=[np.nan]*len(idx))
        agent_exit.name = 'ExitTime'
    else:
        agent_exit = agent_exit.reindex(index=idx)
    df = pd.merge(df, agent_exit.reset_index(), on=mergeon)
    decom_time = series[7]
    if decom_time is not None:
        df = tools.merge_and_fillna_col(df, decom_time.reset_index(), 
                                        'ExitTime', 'DecomTime', on=mergeon)
    duration = series[8]
    df = tools.merge_and_fillna_col(df, duration.reset_index(), 
                                    'ExitTime', 'Duration', on=['SimId'])
    return df

del _agentsdeps, _agentsschema



