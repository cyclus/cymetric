""" Convert able to the default unit system.
"""
import inspect

try:
    from cymetric import schemas
    from cymetric import tools
    from cymetric import evaluator

except ImportError:
    # some wacky CI paths prevent absolute importing, try relative
    from . import schemas
    from . import tools
    from . import evaluator

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

def _gen_norm_metricclass(f, name, r_name, r_regitry, depends, scheme):
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
        raw_unit_registry = r_regitry
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
            return f(self.raw_name, self.raw_unit_registry, *frames)

    Cls.__name__ = str(name)
    evaluator.register_metric(Cls)
    return Cls



def norm_metric(name=None, raw_name=NotImplemented, raw_unit_registry=NotImplemented, depends=NotImplemented, schema=NotImplemented):
    """Decorator that creates metric class from a function or class."""
    def dec(f):
        clsname = name or f.__name__
        return _gen_norm_metricclass(f=f, name=clsname, r_name=raw_name, r_regitry=raw_unit_registry, scheme=schema, depends=depends)
    return dec
             

def build_conversion_col(col):
    conversion_col = [ureg.parse_expression(
        x).to_root_units().magnitude for x in col]
    default_unit = ureg.parse_expression(col[0]).to_root_units().units
    return conversion_col, default_unit


def build_normalized_schema(raw_cls, unit_registry):
    if raw_cls.schema is None:
        return None
    # initialize the normed metric schema
    norm_schema = raw_cls.schema
    # removing units columns form the new schema
    for key in unit_registry:
        idx = norm_schema.index( (unit_registry[key][0], 4, None)) 
        norm_schema.pop(idx)
    return norm_schema


def build_normalized_metric(raw_metric):

    _norm_deps = [raw_metric.__name__]

    _norm_schema = build_normalized_schema(raw_metric, raw_metric.registry)
    _norm_name = "norm_" + raw_metric.__name__
    _raw_name = raw_metric.__name__
    _raw_units_registry = raw_metric.registry

    @norm_metric(name=_norm_name, raw_name=_raw_name, raw_unit_registry=_raw_units_registry, depends=_norm_deps, schema=_norm_schema)
    def new_norm_metric(raw_name, unit_registry, raw):

        norm_pdf = raw.copy(deep=True)
        for unit in unit_registry:
            u_col_name = unit_registry[unit][0]
            u_def_unit = unit_registry[unit][1]
            def_unit = "" 
            # if a column for unit exist parse the colunm convert the value
            # drop the column
            if ( u_col_name != ""):
                conv, def_unit = build_conversion_col(raw[u_col_name]) 
                norm_pdf[unit] *= conv
                norm_pdf.drop([u_col_name], axis=1, inplace=True)
            else: # else use the default unit to convert it
                conv = ureg.parse_expression(u_def_unit).to_root_units().magnitude
                def_unit = ureg.parse_expression(u_def_unit).to_root_units().units
                norm_pdf[unit] *= conv
        norm_pdf.rename(inplace=True, columns={unit : '{0} [{1:~P}]'.format(unit, def_unit)})

        return norm_pdf

    del _norm_deps, _norm_schema, _norm_name


