"""An evaluation context for metrics.
"""
from __future__ import unicode_literals, print_function

import pandas as pd
from cyclus import lib

from cymetric.tools import raw_to_series
from cymetric import units

METRIC_REGISTRY = {}

def register_metric(cls):
    """Adds a metric to the registry."""
    METRIC_REGISTRY[cls.__name__] = cls
    if cls.registry and cls.registry is not NotImplemented:
        units.build_normalized_raw_metric(cls)

class Evaluator(object):
    """An evaluation context for metrics."""

    def __init__(self, db, write=True, normed=True):
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
        self.set_norm = normed

    def get_metric(self, metric, normed=False):
        """Checks if metric is already in the registry; adds it if not."""
        normed_name = "norm_" + metric
        if normed and normed_name in METRIC_REGISTRY:
            metric = normed_name
        if metric not in self.metrics:
            self.metrics[metric] = METRIC_REGISTRY[metric](self.db)
        return self.metrics[metric]

    def eval(self, metric, conds=None, normed=None):
        """Evalutes a metric with the given conditions."""
        requested_metric = metric
        normed_name = "norm_" + metric
        if (normed == True or (normed is None and self.set_norm == True)) and normed_name in METRIC_REGISTRY:
            metric = normed_name
        
        rawkey = (metric, conds if conds is None else frozenset(conds))
        if rawkey in self.rawcache:
            return self.rawcache[rawkey]
        m = self.get_metric(metric, normed)
        frames = []
        for dep in m.dependencies:
            #normed condition avoid inceptions
            frame = self.eval(dep, conds=conds, normed=(dep!=requested_metric) )
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



