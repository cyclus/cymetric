"""A collection of metrics that come stock with cymetric.
"""
from __future__ import print_function, unicode_literals

from cymetric import cyclus
from cymetric.evaluator import register_metric


class Metric(object):

    dependencies = NotImplemented
    schema = NotImplemented
    
    def __init__(self, db):
        self.db = db

def _genmetricclass(f, name, depends, schema):
    class Cls(Metric):
        __name__ = name

        depends = depends
        schema = schema

        def __init__(self, db):
            super(Cls, self).__init__(db)

        def __call__(self, *args, **kwargs):
            # FIXME test if I already exist in the db, read in if I do
            return f()

    register_metric(Cls)
    return Cls


def metric(name=None, depends=NotImplemented, schema=NotImplemented):
    def dec(f):
        if name is None:
            name = f.__name__
        return _genrootclass(f=f, name=name, schema=schema, depends=depends)
    return dec


_matdeps = (('Resources', ('SimId', 'QualId', 'TimeCreated'), 'Quantity'),
            ('Compositions', ('SimId', 'QualId', 'NucId'), 'MassFrac'))

_matschema = (('SimID', cyclus.UUID), ('QualId', cyclus.INT), 
              ('TimeCreated', cyclus.INT), ('NucId', cyclus.INT), 
              ('Mass', cyclus.DOUBLE))

@metric(name='Materials', depends=_matdeps, schema=_matschema)
def materials(series):
    return []


