"""General cymetric tools.
"""
from __future__ import unicode_literals, print_function
import os
import sys

import numpy as np
import pandas as pd

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


def raw_to_series(df, idx, val):
    """Convert data frame to series with multi-index."""
    d = df.set_index(list(map(str, idx)))
    s = df[val].copy()
    s.index = d.index
    return s


def merge_and_fillna_col(left, right, lcol, rcol, how='inner', on=None):
    """Merges two dataframes and fills the values of the left column
    with the values from the right column. A copy of left is returned.
    
    Parameters
    ----------
    left : pd.DataFrame
        The left data frame
    right : pd.DataFrame
        The right data frame
    lcol : str
        The left column name
    rcol : str
        The right column name
    how : str, optional
        How to perform merge, same as in pd.merge()
    on : list of str, optional
        Which columns to merge on, same as in pd.merge()
    """
    m = pd.merge(left, right, how=how, on=on)
    f = m[lcol].fillna(m[rcol])
    left[lcol] = f
    return left


def ensure_dt_bytes(dt):
    """Ensures that a structured numpy dtype is given in a Python 2 & 3
    compatible way.
    """
    if sys.version_info[0] > 2:
        return dt
    dety = []
    for t in dt:
        t0 = t[0].encode() if isinstance(t[0], unicode) else t[0]
        t1 = t[1].encode() if isinstance(t[1], unicode) else t[1]
        ty = (t0, t1)
        if len(t) == 3:
            ty = ty + t[2:]
        dety.append(ty)
    return dety

def raise_no_pyne(msg, have_pyne=False):
    """Raise an error when PyNE cannot be found."""
    if not have_pyne:
        raise ImportError('pyne could not be imported: ' + msg)
