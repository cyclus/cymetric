""" Convert able to the default unit system.
"""
import inspect

try:
    from cymetric import schemas
    from cymetric import tools
    from cymetric import evaluator
#    from cymetric.metrics import Metric, metric
#    from cymetric.root_metrics import root_metric

except ImportError:
    # some wacky CI paths prevent absolute importing, try relative
    from . import schemas
    from . import tools
    from . import evaluator
#    from .root_metrics import root_metric
    from .metrics import Metric, metric

import pint

ureg = pint.UnitRegistry()

class NormMetric(object):
    """Metric class"""
    dependencies = NotImplemented
    schema = NotImplemented
    registry = NotImplemented

    def __init__(self, db):
        self.db = db

    @property
    def name(self):
        return self.__class__.__name__

def _gen_norm_metricclass(f, name, r_name, depends, scheme):
    """Creates a new metric class with a given name, dependencies, and schema.

    Parameters
    ----------
    name : str
        Metric name
    depends : list of lists (table name, tuple of indices, column name)
        Dependencies on other database tables (metrics or root metrics)
    scheme : list of tuples (column name, data type)
        Schema for metric
    """
    if not isinstance(scheme, schemas.schema):
        scheme = schemas.schema(scheme)

    class Cls(NormMetric):
        dependencies = depends
        schema = scheme
        func = staticmethod(f)
        raw_name = r_name
        __doc__ = inspect.getdoc(f)

        def __init__(self, db):
            """Constructor for metric object in database."""
            super(Cls, self).__init__(db)

        def __call__(self, frames, conds=None, known_tables=None, *args, **kwargs):
            """Computes metric for given input data and conditions."""
            # FIXME test if I already exist in the db, read in if I do
            if known_tables is None:
                known_tables = self.db.tables()
            if self.name in known_tables:
                return self.db.query(self.name, conds=conds)
            return f(self.raw_name, *frames)

    Cls.__name__ = str(name)
    evaluator.register_metric(Cls)
    return Cls



def norm_metric(name=None, raw_name=NotImplemented, depends=NotImplemented, schema=NotImplemented):
    """Decorator that creates metric class from a function or class."""
    def dec(f):
        clsname = name or f.__name__
        return _gen_norm_metricclass(f=f, name=clsname, r_name=raw_name, scheme=schema, depends=depends)
    return dec
             

def build_conversion_col(col):
    conversion_col = [ureg.parse_expression(
        x).to_root_units().magnitude for x in col]
    return conversion_col


def build_normalized_schema(raw_cls):
    # if not in the unit registery: nothing to do
    if (raw_cls.__name__ not in evaluator.UNITS_REGISTRY):
        return raw_cls.schema

    # initialize the normed metric schema
    norm_schema = raw_cls.schema

    # removing units columns form the new schema
    unit_registry = evaluator.UNITS_REGISTRY[raw_cls.__name__]
    print(norm_schema)
    for key in unit_registry:
        idx = norm_schema.index( (unit_registry[key][0], 4, None)) 
        norm_schema.pop(idx)
    print(norm_schema)
    return norm_schema


def build_normalized_metric(raw_metric):

    _norm_deps = [raw_metric.__name__]

    _norm_schema = build_normalized_schema(raw_metric)
    _norm_name = "norm_" + raw_metric.__name__
    _raw_name = raw_metric.__name__

    @norm_metric(name=_norm_name, raw_name=_raw_name , depends=_norm_deps, schema=_norm_schema)
    def new_norm_metric(raw_name, raw):
        if (raw_name not in evaluator.UNITS_REGISTRY):
             return raw

        unit_registry = evaluator.UNITS_REGISTRY[raw_name]
        norm_pdf = raw.copy(deep=True)
        for unit in unit_registry:
            u_col_name = unit_registry[unit][0]
            u_def_unit = unit_registry[unit][1]
            
            # if a column for unit exist parse the colunm convert the value
            # drop the column
            if ( u_col_name != ""):
                conv = build_conversion_col(raw[u_col_name]) 
                norm_pdf[unit] *= conv
                norm_pdf.drop([u_col_name], axis=1, inplace=True)
            else: # else use the default unit to convert it
                conv = ureg.parse_expression(u_def_unit).to_root_units().magnitude
                norm_pdf[unit] *= conv

        return norm_pdf

    del _norm_deps, _norm_schema, _norm_name


