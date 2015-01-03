"""General cymetric tools.
"""
from __future__ import unicode_literals, print_function
import os

from cymetric import cyclus


EXT_BACKENDS = {'.h5': cyclus.Hdf5Back, '.sqlite': cyclus.SqliteBack}

def dbopen(fname):
    """Opens a Cyclus database."""
    _, ext = os.path.splitext(fname)
    if ext not in EXT_BACKENDS:
        msg = ('The backend database type of {0!r} could not be determined from '
               'extension {1!r}.')
        raise ValueError(msg.format(fname, ext))
    db = EXT_BACKENDS[ext](fname)
    return db
