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

@dbtest
def test_convint_gettransactiondf(db,fname,backend):
    #db = cym.dbopen()
    myEval = cym.Evaluator(db)
    cal = com.get_transaction_df(myEval)

    exp_head = ['SimId', 'ReceiverId', 'ReceiverProto',
                'SenderId', 'SenderProto', 'TransactionId',
                'ResourceId', 'Commodity', 'Time']

    assert_equal(list(cal), exp_head) # CHeck we have the correct headers

    cal = cal.drop('SimId', 1) # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1) # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1) # SimId change at each test need to drop it

    refs = pd.DataFrame(np.array([
        (15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
        (17, 'Reactor3', 13, 'UOX_Source', 'uox', 3),
        (17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
    ], dtype = ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
        ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)
    
    
    # test single sender
    cal = com.get_transaction_df(myEval,send_list=['UOX_Source'])

    cal = cal.drop('SimId', 1) # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1) # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1) # SimId change at each test need to drop it

    refs = pd.DataFrame(np.array([
        (15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
        (17, 'Reactor3', 13, 'UOX_Source', 'uox', 3),
    ], dtype = ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
        ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)

    
    # test multiple sender
    cal = com.get_transaction_df(myEval,send_list=['UOX_Source', 'MOX_Source'])
    
    cal = cal.drop('SimId', 1) # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1) # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1) # SimId change at each test need to drop it

    refs = pd.DataFrame(np.array([
        (15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
        (17, 'Reactor3', 13, 'UOX_Source', 'uox', 3),
        (17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
    ], dtype = ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
        ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)

    
    # test single receiver
    cal = com.get_transaction_df(myEval, rec_list=['Reactor1'])
    
    cal = cal.drop('SimId', 1) # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1) # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1) # SimId change at each test need to drop it

    refs = pd.DataFrame(np.array([
        (15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
    ], dtype = ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
        ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)

    
    # test multiple sender
    cal = com.get_transaction_df(myEval, rec_list=['Reactor1', 'Reactor3'])
    cal = cal.drop('SimId', 1) # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1) # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1) # SimId change at each test need to drop it

    refs = pd.DataFrame(np.array([
        (15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (17, 'Reactor3', 13, 'UOX_Source', 'uox', 3),
        (17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
    ], dtype = ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
        ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)


# test multiple sender and multiple receiver
    cal = com.get_transaction_df(myEval,send_list=['UOX_Source', 'MOX_Source'],
            rec_list=['Reactor1', 'Reactor2'])
    cal = cal.drop('SimId', 1) # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1) # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1) # SimId change at each test need to drop it

    refs = pd.DataFrame(np.array([
        (15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
    ], dtype = ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
        ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)


if __name__ == "__main__":
    nose.runmodule()











#[left]:  [6, 0, 2, 5, 1, 4, 8, 3, 7]
#[left]:  [6, 0, 1, 4, 2, 5, 7, 3, 8]
#[right]: [6, 0, 2, 5, 1, 4, 8, 3, 7]

#@dbtest
#def test_resources(db, fname, backend):
#    r = root_metrics.resources(db=db)
#    obs = r()
#    assert_less(0, len(obs))
#    assert_equal('Resources', r.name)
