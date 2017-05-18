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
def test_convint_get_transaction_df(db, fname, backend):
    myEval = cym.Evaluator(db)
    cal = com.get_transaction_df(myEval)

    exp_head = ['SimId', 'ReceiverId', 'ReceiverProto', 'SenderId',
                'SenderProto', 'TransactionId', 'ResourceId', 'Commodity', 'Time']

    assert_equal(list(cal), exp_head)  # CHeck we have the correct headers

    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

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
    ], dtype=ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)

    # test single sender
    cal = com.get_transaction_df(myEval, send_list=['UOX_Source'])

    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

    refs = pd.DataFrame(np.array([
        (15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
        (17, 'Reactor3', 13, 'UOX_Source', 'uox', 3),
    ], dtype=ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)

    # test multiple sender
    cal = com.get_transaction_df(
        myEval, send_list=['UOX_Source', 'MOX_Source'])

    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

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
    ], dtype=ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)

    # test single receiver
    cal = com.get_transaction_df(myEval, rec_list=['Reactor1'])

    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

    refs = pd.DataFrame(np.array([
        (15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
    ], dtype=ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)

    # test multiple sender
    cal = com.get_transaction_df(myEval, rec_list=['Reactor1', 'Reactor3'])
    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

    refs = pd.DataFrame(np.array([
        (15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (17, 'Reactor3', 13, 'UOX_Source', 'uox', 3),
        (17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
    ], dtype=ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)


# test multiple sender and multiple receiver
    cal = com.get_transaction_df(myEval, send_list=['UOX_Source', 'MOX_Source'],
                                 rec_list=['Reactor1', 'Reactor2'])
    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

    refs = pd.DataFrame(np.array([
        (15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
    ], dtype=ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)

    # test single commodity
    cal = com.get_transaction_df(myEval, commod_list=['uox'])
    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

    refs = pd.DataFrame(np.array([
        (15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
        (17, 'Reactor3', 13, 'UOX_Source', 'uox', 3),
    ], dtype=ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)

    # test multiple sender
    cal = com.get_transaction_df(myEval, commod_list=['uox', 'mox'])
    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

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
    ], dtype=ensure_dt_bytes([
        ('ReceiverId', '<i8'), ('ReceiverProto', 'O'), ('SenderId', '<i8'),
        ('SenderProto', 'O'), ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)


@dbtest
def test_convint_get_transaction_nuc_df(db, fname, backend):
    myEval = cym.Evaluator(db)
    cal = com.get_transaction_nuc_df(myEval)

    exp_head = ['SimId', 'ResourceId', 'NucId', 'Mass', 'ReceiverId', 'ReceiverProto',
                'SenderId', 'SenderProto', 'TransactionId', 'Commodity', 'Time']

    assert_equal(list(cal), exp_head)  # CHeck we have the correct headers

    # test single nuclide selection
    cal = com.get_transaction_nuc_df(myEval, nuc_list=['942390000'])
    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

    refs = pd.DataFrame(np.array([
        (942390000, 0.0444814879803, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (942390000, 0.0444814879803, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (942390000, 0.0444814879803, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (942390000, 0.0444814879803, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (942390000, 0.0444814879803, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (942390000, 0.0444814879803, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
        (942390000, 0.0444814879803, 17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
    ], dtype=ensure_dt_bytes([
        ('NucId', '<i8'), ('Mass', '<f8'), ('ReceiverId', '<i8'),
        ('ReceiverProto', 'O'), ('SenderId', '<i8'), ('SenderProto', 'O'),
        ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    #refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)

    # test multiple nuclide selection
    cal = com.get_transaction_nuc_df(
        myEval, nuc_list=['942390000', '922380000'])
    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

    refs = pd.DataFrame(np.array([
        (922380000, 0.7872433760310, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (942390000, 0.0444814879803, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (922380000, 0.7872433760310, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (942390000, 0.0444814879803, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (922380000, 0.7872433760310, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (942390000, 0.0444814879803, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (922380000, 0.7872433760310, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (942390000, 0.0444814879803, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (922380000, 0.7872433760310, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (942390000, 0.0444814879803, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (922380000, 0.7872433760310, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
        (942390000, 0.0444814879803, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
        (922380000, 0.7872433760310, 17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
        (942390000, 0.0444814879803, 17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
        (922380000, 0.9600000000000, 17, 'Reactor3', 13, 'UOX_Source', 'uox', 3),
        (922380000, 0.9600000000000, 15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
    ], dtype=ensure_dt_bytes([
        ('NucId', '<i8'), ('Mass', '<f8'), ('ReceiverId', '<i8'),
        ('ReceiverProto', 'O'), ('SenderId', '<i8'), ('SenderProto', 'O'),
        ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    #refs.index = refs.index.astype('str')
    assert_frame_equal(cal, refs)


@dbtest
def test_convint_get_transaction_activity_df(db, fname, backend):
    myEval = cym.Evaluator(db)
    cal = com.get_transaction_activity_df(myEval)

    exp_head = ['SimId', 'ResourceId', 'NucId', 'Activity', 'ReceiverId', 'ReceiverProto',
                'SenderId', 'SenderProto', 'TransactionId', 'Commodity', 'Time']

    assert_equal(list(cal), exp_head)  # CHeck we have the correct headers

    # test single nuclide selection
    cal = com.get_transaction_activity_df(myEval, nuc_list=['942390000'])
    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

    refs = pd.DataFrame(np.array([
        (942390000, 102084984531.0, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (942390000, 102084984531.0, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (942390000, 102084984531.0, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (942390000, 102084984531.0, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (942390000, 102084984531.0, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (942390000, 102084984531.0, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
        (942390000, 102084984531.0, 17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
    ], dtype=ensure_dt_bytes([
        ('NucId', '<i8'), ('Activity', '<f8'), ('ReceiverId','<i8'),
        ('ReceiverProto', 'O'), ('SenderId', '<i8'), ('SenderProto', 'O'),
        ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    assert_frame_equal(cal, refs)

    # test multiple nuclide selection
    cal = com.get_transaction_activity_df(
        myEval, nuc_list=['942390000', '922380000'])
    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

    refs = pd.DataFrame(np.array([
        (922380000, 9790360.331530, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (942390000, 102084984531.0, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (922380000, 9790360.331530, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (942390000, 102084984531.0, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (922380000, 9790360.331530, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (942390000, 102084984531.0, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (922380000, 9790360.331530, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (942390000, 102084984531.0, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (922380000, 9790360.331530, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (942390000, 102084984531.0, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (922380000, 9790360.331530, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
        (942390000, 102084984531.0, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
        (922380000, 9790360.331530, 17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
        (942390000, 102084984531.0, 17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
        (922380000, 11938805.97080, 17, 'Reactor3', 13, 'UOX_Source', 'uox', 3),
        (922380000, 11938805.97080, 15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
    ], dtype=ensure_dt_bytes([
        ('NucId', '<i8'), ('Activity', '<f8'), ('ReceiverId','<i8'),
        ('ReceiverProto', 'O'), ('SenderId', '<i8'), ('SenderProto', 'O'),
        ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    assert_frame_equal(cal, refs)


@dbtest
def test_convint_get_transaction_decayheat_df(db, fname, backend):
    myEval = cym.Evaluator(db)
    cal = com.get_transaction_decayheat_df(myEval)

    exp_head = ['SimId', 'ResourceId', 'NucId', 'DecayHeat', 'ReceiverId', 'ReceiverProto',
                'SenderId', 'SenderProto', 'TransactionId', 'Commodity', 'Time']

    assert_equal(list(cal), exp_head)  # CHeck we have the correct headers

    # test single nuclide selection
    cal = com.get_transaction_decayheat_df(myEval, nuc_list=['942390000'])
    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

    refs = pd.DataFrame(np.array([
        (942390000, 3.34065303191e+30, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (942390000, 3.34065303191e+30, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (942390000, 3.34065303191e+30, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (942390000, 3.34065303191e+30, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (942390000, 3.34065303191e+30, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (942390000, 3.34065303191e+30, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
        (942390000, 3.34065303191e+30, 17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
    ], dtype=ensure_dt_bytes([
        ('NucId', '<i8'), ('DecayHeat', '<f8'), ('ReceiverId','<i8'),
        ('ReceiverProto', 'O'), ('SenderId', '<i8'), ('SenderProto', 'O'),
        ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    assert_frame_equal(cal, refs)

    # test multiple nuclide selection
    cal = com.get_transaction_decayheat_df(
        myEval, nuc_list=['942390000', '922380000'])
    cal = cal.drop('SimId', 1)  # SimId change at each test need to drop it
    # SimId change at each test need to drop it
    cal = cal.drop('TransactionId', 1)
    # SimId change at each test need to drop it
    cal = cal.drop('ResourceId', 1)

    refs = pd.DataFrame(np.array([
        (922380000, 2.609253035160e26, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (942390000, 3.34065303191e+30, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 1),
        (922380000, 2.609253035160e26, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (942390000, 3.34065303191e+30, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 2),
        (922380000, 2.609253035160e26, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (942390000, 3.34065303191e+30, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 2),
        (922380000, 2.609253035160e26, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (942390000, 3.34065303191e+30, 15, 'Reactor1', 14, 'MOX_Source', 'mox', 3),
        (922380000, 2.609253035160e26, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (942390000, 3.34065303191e+30, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 3),
        (922380000, 2.609253035160e26, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
        (942390000, 3.34065303191e+30, 16, 'Reactor2', 14, 'MOX_Source', 'mox', 4),
        (922380000, 2.609253035160e26, 17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
        (942390000, 3.34065303191e+30, 17, 'Reactor3', 14, 'MOX_Source', 'mox', 4),
        (922380000, 3.18184057182e+26, 17, 'Reactor3', 13, 'UOX_Source', 'uox', 3),
        (922380000, 3.18184057182e+26, 15, 'Reactor1', 13, 'UOX_Source', 'uox', 4),
    ], dtype=ensure_dt_bytes([
        ('NucId', '<i8'), ('DecayHeat', '<f8'), ('ReceiverId','<i8'),
        ('ReceiverProto', 'O'), ('SenderId', '<i8'), ('SenderProto', 'O'),
        ('Commodity', 'O'), ('Time', '<i8')
    ]))
    )
    assert_frame_equal(cal, refs)


@dbtest
def test_convint_get_transaction_timeserie(db, fname, backend):
    myEval = cym.Evaluator(db)
    cal = com.get_transaction_timeseries(myEval)

    exp_head = ['Time', 'Mass']

    assert_equal(list(cal), exp_head)  # CHeck we have the correct headers

    # test single nuclide selection
    cal = com.get_transaction_timeseries(myEval, nuc_list=['942390000'])
    refs = pd.DataFrame(np.array([
        (0, 0.000000000),
        (1, 0.0444814879803),
        (2, 0.0889629759607),
        (3, 0.0889629759607),
        (4, 0.0889629759607),
    ], dtype=ensure_dt_bytes([
        ('Time', '<i8'), ('Mass', '<f8')
    ]))
    )
    assert_frame_equal(cal, refs)

    # test multiple nuclide selection
    cal = com.get_transaction_timeseries(
        myEval, nuc_list=['942390000', '922380000'])
    refs = pd.DataFrame(np.array([
        (0, 0.000000000),
        (1, 0.831724864011),
        (2, 1.66344972802),
        (3, 2.62344972802),
        (4, 2.62344972802),
    ], dtype=ensure_dt_bytes([
        ('Time', '<i8'), ('Mass', '<f8')
    ]))
    )

    assert_frame_equal(cal, refs)


@dbtest
def test_convint_get_transaction_activity_timeserie(db, fname, backend):
    myEval = cym.Evaluator(db)
    cal = com.get_transaction_activity_timeseries(myEval)

    exp_head = ['Time', 'Activity']

    assert_equal(list(cal), exp_head)  # CHeck we have the correct headers

    # test single nuclide selection
    cal = com.get_transaction_activity_timeseries(
        myEval, nuc_list=['942390000'])
    refs = pd.DataFrame(np.array([
        (0, 0.000000000),
        (1, 102084984531.0),
        (2, 204169969062.0),
        (3, 204169969062.0),
        (4, 204169969062.0),
    ], dtype=ensure_dt_bytes([
        ('Time', '<i8'), ('Activity', '<f8')
    ]))
    )
    assert_frame_equal(cal, refs)

    # test multiple nuclide selection
    cal = com.get_transaction_activity_timeseries(
        myEval, nuc_list=['942390000', '922380000'])
    refs = pd.DataFrame(np.array([
        (0, 0.000000000),
        (1, 102094774891.0),
        (2, 204189549782.0),
        (3, 204201488588.0),
        (4, 204201488588.0),
    ], dtype=ensure_dt_bytes([
        ('Time', '<i8'), ('Activity', '<f8')
    ]))
    )

    assert_frame_equal(cal, refs)


if __name__ == "__main__":
    nose.runmodule()
