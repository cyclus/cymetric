"""A collection of metrics that come stock with cymetric.
"""
from __future__ import print_function, unicode_literals

import pandas as pd

from cymetric import cyclus
from cymetric.evaluator import register_metric


class Metric(object):

    dependencies = NotImplemented
    schema = NotImplemented
    
    def __init__(self, db):
        self.db = db

    @property
    def name(self):
        return self.__class__.__name__


def _genmetricclass(f, name, depends, scheme):
    class Cls(Metric):
        dependencies = depends
        schema = scheme

        def __init__(self, db):
            super(Cls, self).__init__(db)

        def __call__(self, series, *args, **kwargs):
            # FIXME test if I already exist in the db, read in if I do
            return f(series)

    Cls.__name__ = str(name)
    register_metric(Cls)
    return Cls


def metric(name=None, depends=NotImplemented, schema=NotImplemented):
    def dec(f):
        clsname = name or f.__name__
        return _genmetricclass(f=f, name=clsname, scheme=schema, depends=depends)
    return dec


_matdeps = (('Resources', ('SimId', 'QualId', 'ResourceId', 'ObjId', 'TimeCreated'), 
                'Quantity'),
            ('Compositions', ('SimId', 'QualId', 'NucId'), 'MassFrac'))

_matschema = (('SimId', cyclus.UUID), ('QualId', cyclus.INT), 
              ('ResourceId', cyclus.INT), ('ObjId', cyclus.INT), 
              ('TimeCreated', cyclus.INT), ('NucId', cyclus.INT), 
              ('Mass', cyclus.DOUBLE))

@metric(name='Materials', depends=_matdeps, schema=_matschema)
def materials(series):
    x = pd.merge(series[0].reset_index(), series[1].reset_index(), 
            on=['SimId', 'QualId'], how='inner').set_index(['SimId', 'QualId', 
                'ResourceId', 'ObjId','TimeCreated', 'NucId'])
    y = x['Quantity'] * x['MassFrac']
    y.name = 'Mass'
    z = y.reset_index()
    return z


