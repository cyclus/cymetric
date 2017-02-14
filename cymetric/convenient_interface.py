#!/usr/bin/env python

import pandas as pd
import numpy as np
import cymetric as cym
import warnings
from cymetric import tools

try:
    from pyne import data
    import pyne.enrichment as enr
    from pyne import nucname
    HAVE_PYNE = True
except ImportError:
    HAVE_PYNE = False


def format_nuclist(nuc_list):
    tools.raise_no_pyne('Unable to format nuclide list!', HAVE_PYNE)
    
    for i in range(len(nuc_list)):
        nuc_list[i] = nucname.id(nuc_list[i])

    return nuc_list


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


def get_transaction_pdf(evaler, send_list=[], rec_list=[], commod_list=[]):
    """
    Filter the Transaction Data Frame on specific sending facility and
    receving facility.

    Parameters
    ----------
    evaler : evaler
    send_list : list of the sending facility
    rec_list : list of the receiving facility
    commod_list : list of the commodity exchanged
    """

    # initiate evaluation
    trans = evaler.eval('Transactions')
    agents = evaler.eval('AgentEntry')

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
        rdc_list.append(['SenderId', send_agent['SenderId'].tolist()])
        if len(commod_list) != 0:
            rdc_list.append(['Commodity', commod_list])

        trans = get_reduced_pdf(trans, rdc_list)

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

    return trans


def get_transaction_nuc_pdf(evaler, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Shape the reduced transation Dta Frame into a simple time serie. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    send_list : list of the sending facility
    rec_list : list of the receiving facility
    commod_list : list of the receiving facility
    nuc_list : list of nuclide to select.
    """

    df = get_transaction_pdf(evaler, send_list, rec_list, commod_list)

    compo = evaler.eval('Materials')
    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)
        compo = get_reduced_pdf(compo, [['NucId', nuc_list]])

    base_col = ['SimId', 'ResourceId']
    added_col = base_col + ['NucId', 'Mass']
    df = merge_n_drop(df, base_col, compo, added_col)

    return df


def get_transaction_activity_pdf(evaler, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Return the transation pdf, with the activities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    send_list : list of the sending facility
    rec_list : list of the receiving facility
    commod_list : list of the receiving facility
    nuc_list : list of nuclide to select.
    """

    df = get_transaction_pdf(evaler, send_list, rec_list, commod_list)

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    compo = evaler.eval('Activity')
    compo = get_reduced_pdf(compo, [['NucId', nuc_list]])

    base_col = ['SimId', 'ResourceId']
    added_col = base_col + ['NucId', 'Activity']
    df = merge_n_drop(df, base_col, compo, added_col)

    return df


def get_transaction_decayheat_pdf(evaler, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Return the transation pdf, with the decayheat. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    send_list : list of the sending facility
    rec_list : list of the receiving facility
    commod_list : list of the receiving facility
    nuc_list : list of nuclide to select.
    """

    df = get_transaction_pdf(evaler, send_list, rec_list, commod_list)

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    compo = evaler.eval('DecayHeat')
    compo = get_reduced_pdf(compo, [['NucId', nuc_list]])

    base_col = ['SimId', 'ResourceId']
    added_col = base_col + ['NucId', 'DecayHeat']
    df = merge_n_drop(df, base_col, compo, added_col)

    return df


def get_transaction_timeseries(evaler, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Shape the reduced transation Dta Frame into a simple time serie. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    send_list : list of the sending facility
    rec_list : list of the receiving facility
    commod_list : list of the receiving facility
    nuc_list : list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    df = get_transaction_nuc_pdf(evaler, send_list, rec_list, commod_list, nuc_list)

    group_end = ['ReceiverProto', 'SenderProto', 'Time']
    group_start = group_end + ['Mass']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    df = df[['Time', 'Mass']].groupby(['Time']).sum()
    df.reset_index(inplace=True)

    return df


def get_transaction_activity_timeseries(evaler, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Shape the reduced transation Dta Frame into a simple time serie. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    send_list : list of the sending facility
    rec_list : list of the receiving facility
    commod_list : list of the receiving facility
    nuc_list : list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)
    
    df = get_transaction_activity_pdf(evaler, send_list, rec_list, commod_list,
            nuc_list)

    group_end = ['ReceiverProto', 'SenderProto', 'Time']
    group_start = group_end + ['Activity']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    df = df[['Time', 'Activity']].groupby(['Time']).sum()
    df.reset_index(inplace=True)

    return df

def get_transaction_decayheat_timeseries(evaler, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Shape the reduced transation Dta Frame into a simple time serie. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    send_list : list of the sending facility
    rec_list : list of the receiving facility
    commod_list : list of the receiving facility
    nuc_list : list of nuclide to select.
    """
    
    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list) 

    df = get_transaction_decayheat_pdf(evaler, send_list, rec_list, commod_list,
            nuc_list)

    group_end = ['ReceiverProto', 'SenderProto', 'Time']
    group_start = group_end + ['DecayHeat']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    df = df[['Time', 'DecayHeat']].groupby(['Time']).sum()
    df.reset_index(inplace=True)

    return df


def get_inventory_pdf(evaler, fac_list=[], nuc_list=[]):
    """
    Shape the reduced inventory Data Frame. Applying nuclides/facilities selection when required.

    Parameters
    ----------
    evaler : evaler
    fac_name : name of the facility
    nuc_list : list of nuclide to select.
    """

    # Get inventory table
    inv = evaler.eval('ExplicitInventory')
    agents = evaler.eval('AgentEntry')

    rdc_list = []  # because we want to get reed of the nuclide asap
    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)
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
    
    return inv
    

def get_inventory_timeseries(evaler, fac_list=[], nuc_list=[]):
    """
    Shape the reduced inventory Data Frame into a simple time serie. Applying
    nuclides/facilities selection when required.

    Parameters
    ----------
    evaler : evaler
    fac_name : name of the facility
    nuc_list : list of nuclide to select.
    """
    
    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)
    
    inv = get_inventory_pdf(evaler, fac_list, nuc_list)

    group_end = ['Time']
    group_start = group_end + ['Quantity']
    inv = inv[group_start].groupby(group_end).sum()
    inv.reset_index(inplace=True)

    return inv


def get_inventory_activity_pdf(evaler, fac_list=[], nuc_list=[]):
    """
    Get a simple time series of the activity of the inventory in the selcted
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    fac_name : name of the facility
    nuc_list : list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    df = get_inventory_pdf(evaler, fac_list, nuc_list)
    for i, row in df.iterrows():
        val = 1000 * data.N_A * df.ix[i, 'Quantity'] * \
            data.decay_const(int(df.ix[i, 'NucId']))
        df.set_value(i, 'Activity', val)

    return df


def get_inventory_activity_timeseries(evaler, fac_list=[], nuc_list=[]):
    """
    Get a simple time series of the decay heat of the inventory in the selcted
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    fac_name : name of the facility
    nuc_list : list of nuclide to select.
    """
    
    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list) 
    
    activity = get_inventory_activity_pdf(evaler, fac_list, nuc_list)
    group_end = ['Time']
    group_start = group_end + ['Activity']
    activity = activity[group_start].groupby(group_end).sum()
    activity.reset_index(inplace=True)
    
    return activity


def get_inventory_decayheat_pdf(evaler, fac_list=[], nuc_list=[]):
    """
    Get a Inventory PDF including the decay heat of the inventory in the selected
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    fac_name : name of the facility
    nuc_list : list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    df = get_inventory_activity_pdf(evaler, fac_list, nuc_list)
    for i, row in df.iterrows():
        val = data.MeV_per_MJ * \
            df.ix[i, 'Activity'] * data.q_val(int(df.ix[i, 'NucId']))
        df.set_value(i, 'DecayHeat', val)

    return df


def get_inventory_decayheat_timeseries(evaler, fac_list=[], nuc_list=[]):
    """
    Get a simple time series of the decay heat of the inventory in the selcted
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    fac_name : name of the facility
    nuc_list : list of nuclide to select.
    """
    
    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)
    
    decayheat = get_inventory_decayheat_pdf(evaler, fac_list, nuc_list)
    group_end = ['Time']
    group_start = group_end + ['DecayHeat']
    decayheat = decayheat[group_start].groupby(group_end).sum()
    decayheat.reset_index(inplace=True)
    
    return decayheat


def get_power_timeseries(evaler, fac_list=[]):
    """
    Shape the reduced Power Data Frame into a simple time serie. Applying
    facilities selection when required.

    Parameters
    ----------
    evaler : evaler
    fac_list : list of name of the facility
    """

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


def get_deployment_timeseries(evaler, fac_list=[]):
    """
    Get a simple time series with deployment schedule of the selected facilities.

    Parameters
    ----------
    evaler : evaler
    fac_name : name of the facility
    """

    # Get inventory table
    df = evaler.eval('AgentEntry')

    rdc_list = []  # because we want to get reed of the facility asap
    if len(fac_list) != 0:
        df = df[df['Prototype'].isin(fac_list)]
        rdc_list.append(['AgentId', df['AgentId'].tolist()])
    else:
        wng_msg = "no faciity provided"
        warnings.warn(wng_msg, UserWarning)

    # Adding a constante column to easely sum the amount of facilities build per
    # time step
    df = df.assign(Value = lambda x: 1)

    group_end = ['EnterTime']
    group_start = group_end + ['Value']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    return df


def get_retirement_timeseries(evaler, fac_list=[]):
    """
    Get a simple time series with retirement schedule of the selected facilities.

    Parameters
    ----------
    evaler : evaler
    fac_name : name of the facility
    """

    # Get inventory table
    df = evaler.eval('AgentEntry')
    df = df[df['Lifetime'] > 0]
    
    rdc_list = []  # because we want to get reed of the facility asap
    if len(fac_list) != 0:
        df = df[df['Prototype'].isin(fac_list)]
        rdc_list.append(['AgentId', df['AgentId'].tolist()])
    else:
        wng_msg = "no faciity provided"
        warnings.warn(wng_msg, UserWarning)

    # Adding a constante column to easely sum the amount of facilities build per
    # time stepi

    df = df.assign(Value = lambda x: 1)
    
    df['DecomTime'] = df['EnterTime'] + df['Lifetime']

    group_end = ['DecomTime']
    group_start = group_end + ['Value']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    return df


