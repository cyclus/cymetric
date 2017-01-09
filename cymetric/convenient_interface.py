#!/usr/bin/env python

from pyne import nucname
import pandas as pd
import numpy as np
import cymetric as cym
import warnings


def merge_n_drop(pdf, base_col, add_pdf, add_col):
    """
    Merge some additionnal columns fram an additionnal Pandas Data Frame
    onother one and then remove the second base column (keeping SimID
    information).
    Parameters
    ----------
    pdf : Pandas Data Frame
    base_col : list of the base columns names
    add_pdf : Pandas Data Frame to add in the pdf one
    add_col : columns tobe added
    """
    pdf = pd.merge(add_pdf[add_col], pdf, on=base_col)
    pdf.drop(base_col[1], 1)
    return pdf

def get_reduced_pdf(pdf, rdc_list):
    """
    Filter the pdf Pandas Data Frame according to the rdc_list (list of item
    in the corresponding columns).
    Parameters
    ----------
    pdf : Pandas Data Frame
    rdc_list : list of pair of string and string list.
    """
    for rdc in rdc_list:
        if len(rdc[1]) != 0:
            print(rdc[0])
            print(rdc[1])
            pdf = pdf[ pdf[rdc[0]].isin(rdc[1]) ]
        else:
            wng_msg = "Empty list provided for " + rdc[0] + " key."
            warnings.wrn(wng_msg, UserWarning)
    return pdf

def get_reduced__trans_pdf(db, send_name, rec_name):
    """
    Filter the Transaction Data Frame on specific sending facility and
    receving facility.

    Parameters
    ----------
    db : database
    send_name : name of the sending facility ('All' for any)
    rec_name : name of the receiving facility ('All for any)

    Attributes
    ----------
    """

    # initiate evaluation
    evaler = cym.Evaluator(db)
    trans = evaler.eval('Transactions')
    agents = evaler.eval('AgentEntry')
    rsc = evaler.eval('Resources')

    rec_list = agents.rename(index=str, columns={'AgentId': 'ReceiverId'})
    if rec_name != 'All':
        rec_list = rec_list.loc[lambda df: df.Prototype == rec_name,:]
    
    send_list = agents.rename(index=str, columns={'AgentId': 'SenderId'})
    if send_name != 'All':
        send_list = send_list.loc[lambda df: df.Prototype == send_name,:]

    # check if sender and receiver exist
    if rec_list.empty or send_list.empty:
        return None
    else:
        trans = get_reduced_pdf(trans, [
            ['ReceiverId', rec_list['ReceiverId'].tolist() ],
            ['SenderId', send_list['SenderId'].tolist() ] ])
        rsc = rsc[ rsc['ResourceId'].isin( trans.ResourceId ) ]

        base_col = ['SimId', 'SenderId']
        added_col = base_col + ['Prototype']
        trans = merge_n_drop(trans, base_col, send_list, added_col)
        trans = trans.rename(index=str, columns={'Prototype': 'SenderProto'})

        base_col = ['SimId', 'ReceiverId']
        added_col = base_col + ['Prototype']
        trans = merge_n_drop(trans, base_col, send_list, added_col)
        trans = trans.rename(index=str, columns={'Prototype': 'ReceiverProto'})

        base_col = ['SimId', 'RessourceId']
        added_col = base_col + ['QualId', 'Quantity', 'Unit']
        trans = merge_n_drop(trans, base_col, rsc, added_col)

    return trans


def get_transaction_timeseries(db, send_name='All', rec_name='All', nuc_list=[]):
    """
    Shape the reduced transation Dta Frame into a simple time serie. Apply
    some nuclei selection if required.

    Parameters
    ----------
    db : database
    send_name : name of the sending facility
    rec_name : name of the receiving facility
    nuc_list : list of nuclide to select.
    """

    df = get_reduced__trans_pdf(db, send_name, rec_name)

    if len(nuc_list) != 0:
        compo = evaler.eval('Compositions')
        compo = get_reduced_pdf(compo, [['NucId', nuc_list]])

        base_col = ['SimId', 'QualId']
        added_col = base_col + ['NucId', 'MassFrac']
        df = merge_n_drop(df, base_col, compo, added_co)

        df['Quantity'] = df['Quantity'] * df['MassFrac']

        group_end = ['ReceiverProto', 'SenderProto', 'Time']
        group_start = group_end + ['Quantity']
        df = df[group_start].groupby(gourp_end).sum()
    else:
        wng_msg = "no nuclide provided"
        warnings.wrn(wng_msg, UserWarning)

    if sender == 'All':
        grouped_trans = df[['ReceiverProto', 'Time', 'Quantity']].groupby(
            ['ReceiverProto', 'Time']).sum()
        trans_table = grouped_trans.loc[receiver]
    elif receiver == 'All':
        grouped_trans = df[['SenderProto', 'Time', 'Quantity']].groupby(
            ['SenderProto', 'Time']).sum()
        trans_table = grouped_trans.loc[sender]
    else:
        grouped_trans = df[['ReceiverProto', 'SenderProto', 'Time',
                            'Quantity']].groupby(['ReceiverProto', 'SenderProto',
                                                  'Tme']).sum()
        trans_table = grouped_trans.loc[receiver].loc[sender]

    return trans_table


def get_inventory_timeseries(db, fac_name, nuc_list):
    """
    Shape the reduced inventory Dta Frame into a simple time serie. Apply
    some nuclei selection if required.

    Parameters
    ----------
    db : database
    fac_name : name of the facility
    nuc_list : list of nuclide to select.
    """
    evaler = cym.Evaluator(db)

    # Get inventory table
    inv = evaler.eval('ExplicitInventory')

    rdc_list =[] # because we want to get reed of the nuclide asap
    if len(nuc_list) != 0:
        rdc_list.append(['NucId', nuc_list])
    else:
        wng_msg = "no nuclide provided"
        warnings.wrn(wng_msg, UserWarning)
    
    selected_agents = agents.loc[lambda df: df.Prototype == fac_name, :]
    if fac_name != 'All':
        rdc_list.append(['AgentId', agents.ReceiverId])
    else:
        wng_msg = "no faciity provided"
        warnings.wrn(wng_msg, UserWarning)

    inv = get_reduced_pdf(inv, rdc_list) 
    
    base_col = ['SimId', 'AgentId']
    added_col = base_col + ['Prototype']
    inv = merge_n_drop(inv, base_col, send_list, added_col)

    group_end = ['prototype', 'time']
    group_start = group_end + ['Quantity']
    inv = inv[group_start].groupby(gourp_end).sum()

    return inv_table
