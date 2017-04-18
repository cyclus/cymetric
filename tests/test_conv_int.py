"""Tests for convinient interface method"""
from __future__ import print_function, unicode_literals
from uuid import UUID
import os
import subprocess
from functools import wraps

import nose
from nose.tools import assert_equal, assert_less

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal


from tools import setup, dbtest

import cymetric as cym
from cymetric import convenient_interface as com
from cymetric.tools import raw_to_series, ensure_dt_bytes

def test_convint_gettransactiondf():
    db = cym.dbopen(example_path)
    myEval = cym.Evaluator(db)
    cal = com.get_transaction_df(myEval)

    refs = pd.DataFrame(np.array([
        (UUID('4485c97d-59bc-4eda-ad42-63a7f042053a'), 15,
         'Reactor1', 13, 'UOX_Source', 8, 78, 'uox', 4),
        (UUID('4485c97d-59bc-4eda-ad42-63a7f042053a'), 15,
         'Reactor1', 14, 'MOX_Source', 0, 8,  'mox', 1),
        (UUID('4485c97d-59bc-4eda-ad42-63a7f042053a'), 15,
         'Reactor1', 14, 'MOX_Source', 1, 23, 'mox', 2),
        (UUID('4485c97d-59bc-4eda-ad42-63a7f042053a'), 15,
         'Reactor1', 14, 'MOX_Source', 3, 47, 'mox', 3),
        (UUID('4485c97d-59bc-4eda-ad42-63a7f042053a'), 16,
         'Reactor2', 14, 'MOX_Source', 2, 25, 'mox', 2),
        (UUID('4485c97d-59bc-4eda-ad42-63a7f042053a'), 16,
         'Reactor2', 14, 'MOX_Source', 4, 49, 'mox', 3),
        (UUID('4485c97d-59bc-4eda-ad42-63a7f042053a'), 16,
         'Reactor2', 14, 'MOX_Source', 6, 74, 'mox', 4),
        (UUID('4485c97d-59bc-4eda-ad42-63a7f042053a'), 17,
         'Reactor3', 13, 'UOX_Source', 5, 51, 'uox', 3),
        (UUID('4485c97d-59bc-4eda-ad42-63a7f042053a'), 17,
         'Reactor3', 14, 'MOX_Source', 7, 76, 'mox', 4),
    ], dtype = ensure_dt_bytes([
        ('SimId', 'O'), ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), 
        ('SenderId', '<i8'), ('SenderProto', 'O'), ('TransactionId', '<i8'),
        ('ResourceId', '<i8'), ('Commodity', 'O'), ('Time', '<i8')
        ]))
    )

#@dbtest
#def test_resources(db, fname, backend):
#    r = root_metrics.resources(db=db)
#    obs = r()
#    assert_less(0, len(obs))
#    assert_equal('Resources', r.name)


if __name__ == "__main__":
    nose.runmodule()
