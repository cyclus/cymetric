"""An evaluation context for metrics.
"""
from __future__ import unicode_literals, print_function

METRIC_REGISTRY = {}

def register_metric(cls):
    """Adds a metric to the registry."""
    METRIC_REGISTRY[cls.__name__] = cls


class Evaluator(object):
    """An evaluation context for metrics."""

    def __init__(self, db):
        """Parameters
        ----------
        db : database

        Attributes
        ----------
        metrics : dict
            Metric instances bound the evalator's database.
        rawcache : dict
            Results of querying metrics with given conditions.
        """
        self.metrics = {}
        self.rawcache = {}

    def get_metric(self, metric):
        if metric not in self.metrics:
            self.metrics[metric] = METRIC_REGISTRY[metric](db)
        return self.metrics[metric]

    def eval(self, metric, conds=None):
        """Evalutes a metric with the given conditions."""
        rawkey = (metric, conds if conds is None else frozenset(conds))
        if rawkey in self.rawcache:
            return self.rawcache[rawkey]
        m = self.get_metric(metric)
        series = []
        for dep in m.depends:
            d = self.eval(dep[0], conds=conds)
            s = raw_to_series(d, dep[1], dep[2])
            series.append(s)
        raw = m(series)
        self.rawcache[rawkey] = raw
        # FIXME write back to db here
        return raw


def eval(metric, db, conds=None):
    """Evalutes a metric with the given conditions in a database."""
    e = Evaluator(db)
    return e.eval(metric, conds=conds)
