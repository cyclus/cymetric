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
            pdf = pdf[pdf[rdc[0]].isin(rdc[1])]
        else:
            wng_msg = "Empty list provided for " + rdc[0] + " key."
            warnings.warn(wng_msg, UserWarning)
    return pdf


def get_transaction_pdf(db, send_list=[], rec_list=[], commod_list=[]):
    """
    Filter the Transaction Data Frame on specific sending facility and
    receving facility.

    Parameters
    ----------
    db : database
    send_list : list of the sending facility
    rec_list : list of the receiving facility
    commod_list : list of the commodity exchanged
    """

    # initiate evaluation
    evaler = cym.Evaluator(db)
    trans = evaler.eval('Transactions')
    agents = evaler.eval('AgentEntry')
    rsc = evaler.eval('Resources')

    rec_agent = agents.rename(index=str, columns={'AgentId': 'ReceiverId'})
    if len(rec_list) != 0:
        rec_agent = rec_agent[rec_agent['Prototype'].isin(rec_list)]

    send_agent = agents.rename(index=str, columns={'AgentId': 'SenderId'})
    if len(send_list) != 0:
        send_agent = send_agent[send_agent['Prototype'].isin(send_list)]

    # check if sender and receiver exist
    if rec_agent.empty or send_agent.empty:
        return None
    else:
        # Clean Transation PDF
        rdc_list = []
        rdc_list.append(['ReceiverId', rec_agent['ReceiverId'].tolist()])
        rdc_list.append(['SenderId',send_agent['SenderId'].tolist()])
        if len(commod_list) != 0:
            rdc_list.append(['Commodity',commod_list])
            
        trans = get_reduced_pdf(trans, rdc_list )

        # Merge Sender to Transaction PDF
        base_col = ['SimId', 'SenderId']
        added_col = base_col + ['Prototype']
        trans = merge_n_drop(trans, base_col, send_agent, added_col)
        trans = trans.rename(index=str, columns={'Prototype': 'SenderProto'})

        # Merge Receiver to Transaction PDF
        base_col = ['SimId', 'ReceiverId']
        added_col = base_col + ['Prototype']
        trans = merge_n_drop(trans, base_col, rec_agent, added_col)
        trans = trans.rename(index=str, columns={'Prototype': 'ReceiverProto'})

        # Merge Resource to Transaction PDF
        rsc = rsc[rsc['ResourceId'].isin(trans.ResourceId)]
        base_col = ['SimId', 'ResourceId']
        added_col = base_col + ['QualId'] + ['Quantity'] + ['Units']
        trans = merge_n_drop(trans, base_col, rsc, added_col)

    return trans


def get_transaction_timeseries(db, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Shape the reduced transation Dta Frame into a simple time serie. Apply
    some nuclei selection if required.

    Parameters
    ----------
    db : database
    send_list : list of the sending facility
    rec_list : list of the receiving facility
    commod_list : list of the receiving facility
    nuc_list : list of nuclide to select.
    """

    df = get_transaction_pdf(db, send_list, rec_list, commod_list)

    if len(nuc_list) != 0:
        for i in range(len(nuc_list)):
            nuc_list[i] = nucname.id(nuc_list[i])

        evaler = cym.Evaluator(db)
        compo = evaler.eval('Compositions')
        compo = get_reduced_pdf(compo, [['NucId', nuc_list]])

        base_col = ['SimId', 'QualId']
        added_col = base_col + ['NucId', 'MassFrac']
        df = merge_n_drop(df, base_col, compo, added_col)

        df['Quantity'] = df['Quantity'] * df['MassFrac']

        group_end = ['ReceiverProto', 'SenderProto', 'Time']
        group_start = group_end + ['Quantity']
        df = df[group_start].groupby(group_end).sum()
    else:
        wng_msg = "no nuclide provided"
        warnings.warn(wng_msg, UserWarning)
        
    trans = df[['Time', 'Quantity']].groupby(['Time']).sum()
    trans.reset_index(inplace=True)

    return trans


def get_inventory_timeseries(db, fac_list=[], nuc_list=[]):
    """
    Shape the reduced inventory Data Frame into a simple time serie. Apply
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
    agents = evaler.eval('AgentEntry')

    rdc_list = []  # because we want to get reed of the nuclide asap
    if len(nuc_list) != 0:
        for i in range(len(nuc_list)):
            nuc_list[i] = nucname.id(nuc_list[i])
        rdc_list.append(['NucId', nuc_list])
    else:
        wng_msg = "no nuclide provided"
        warnings.warn(wng_msg, UserWarning)

    if len(fac_list) != 0:
        agents = agents[agents['Prototype'].isin(fac_list)]
        rdc_list.append(['AgentId', agents['AgentId'].tolist()])
    else:
        wng_msg = "no faciity provided"
        warnings.warn(wng_msg, UserWarning)
    inv = get_reduced_pdf(inv, rdc_list)

    base_col = ['SimId', 'AgentId']
    added_col = base_col + ['Prototype']
    inv = merge_n_drop(inv, base_col, agents, added_col)
    group_end = ['Time']
    group_start = group_end + ['Quantity']
    inv = inv[group_start].groupby(group_end).sum()
    inv.reset_index(inplace=True)

    return inv


def get_power_timeseries(db, fac_list=[]):
    """
    Shape the reduced Power Data Frame into a simple time serie. Apply
    some facility selection if required.

    Parameters
    ----------
    db : database
    fac_list : list of name of the facility
    """
    evaler = cym.Evaluator(db)

    # Get inventory table
    power = evaler.eval('TimeSeriesPower')
    agents = evaler.eval('AgentEntry')

    rdc_list = []  # because we want to get reed of the facility asap
    print(fac_list)
    if len(fac_list) != 0:
        agents = agents[agents['Prototype'].isin(fac_list)]
        rdc_list.append(['AgentId', agents['AgentId'].tolist()])
    power = get_reduced_pdf(power, rdc_list)

    base_col = ['SimId', 'AgentId']
    added_col = base_col + ['Prototype']
    power = merge_n_drop(power, base_col, agents, added_col)

    group_end = ['Time']
    group_start = group_end + ['Value']
    power = power[group_start].groupby(group_end).sum()
    power.reset_index(inplace=True)

    return power
