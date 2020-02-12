"""An evaluation context for metrics.
"""
from __future__ import unicode_literals, print_function

import pandas as pd
from cyclus import lib

from cymetric.tools import raw_to_series


METRIC_REGISTRY = {}
UNITS_REGISTRY = {}

def register_metric(cls):
    """Adds a metric to the registry."""
    METRIC_REGISTRY[cls.__name__] = cls
    if cls.__name__ not in UNITS_REGISTRY and cls.registry is not None:
        UNITS_REGISTRY[cls.__name__] = cls.registry
        build_normalised_metric(cls)

class Evaluator(object):
    """An evaluation context for metrics."""

    def __init__(self, db, write=True):
        """Parameters
        ----------
        db : database
        write : bool, optional
            Flag for whether metrics should be written to the database.

        Attributes
        ----------
        metrics : dict
            Metric instances bound the evaluator's database.
        rawcache : dict
            Results of querying metrics with given conditions.
        """
        self.write = write
        self.metrics = {}
        self.rawcache = {}
        self.db = db
        self.recorder = rec = lib.Recorder(inject_sim_id=False)
        rec.register_backend(db)
        self.known_tables = db.tables
        #try:
        #    self.registry = self.db.query("table_registry")

    def get_metric(self, metric):
        """Checks if metric is already in the registry; adds it if not."""
        if metric not in self.metrics:
            self.metrics[metric] = METRIC_REGISTRY[metric](self.db)
        return self.metrics[metric]

    def eval(self, metric, conds=None):
        """Evalutes a metric with the given conditions."""
        rawkey = (metric, conds if conds is None else frozenset(conds))
        if rawkey in self.rawcache:
            return self.rawcache[rawkey]
        m = self.get_metric(metric)
        frames = []
        for dep in m.dependencies:
            frame = self.eval(dep, conds=conds)
            frames.append(frame)
        raw = m(frames=frames, conds=conds, known_tables=self.known_tables)
        if raw is None:
            return raw
        self.rawcache[rawkey] = raw
        # write back to db
        if (m.name in self.known_tables) or (not self.write):
            return raw
        rec = self.recorder
        rawd = raw.to_dict(orient='list')
        for i in range(len(raw)):
            d = rec.new_datum(m.name)
            for field, dbtype, shape in m.schema:
                fname = m.schema.byte_names[field]
                val = rawd[str(field)][i]
                d = d.add_val(fname, val, type=dbtype, shape=shape)
            d.record()
        self.known_tables.add(m.name)
        return raw


def eval(metric, db, conds=None, write=True):
    """Evalutes a metric with the given conditions in a database."""
    e = Evaluator(db, write=write)
    return e.eval(str(metric), conds=conds)


def build_normalisd_metric(cls):
    _dsdeps = [cls.__name__]

    _dsschema = build_normalized_schema(cls.schema)
    norm_name = cls.__name__ + "_norm"
    
    @metric(name=norm_name, depends=_dsdeps, schema=_dsschema)
    def normalization(raw_metric):
        return rtn

    del _dsdeps, _dsschema

def build_normalized_schema(raw_cls):
    # if not in the unit registery: nothing to do
    if (raw_cls.__name__ not in UNITS_REGISTRY):
        return raw_cls.schema
    
    # initialize the normed metric schema
    norm_schema = raw_cls.schema
    
    # removing units columns form the new schema 
    unit_registry = UNITS_REGISTRY[raw_cls.__name__]
    for key in unit_registry:
        norm_schema.pop(unit_registry[key][0], None)    
    
    return norm_schema

