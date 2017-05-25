#!/usr/bin/env python

import warnings

import pandas as pd
import numpy as np

import cymetric as cym
from cymetric import tools

try:
    from pyne import data
    import pyne.enrichment as enr
    from pyne import nucname
    HAVE_PYNE = True
except ImportError:
    HAVE_PYNE = False

try:
    from graphviz import Digraph
    HAVE_GRAPHVIZ = True
except ImportError:
    HAVE_GRAPHVIZ = False


def format_nuclist(nuc_list):
    """
    format the nuclide list provided by the users into a standart format:
    ZZAAIIIII.
    Parameters
    ----------
    nuc_list:  list of nuclides
    """
    tools.raise_no_pyne('Unable to format nuclide list!', HAVE_PYNE)

    for i in range(len(nuc_list)):
        nuc_list[i] = nucname.id(nuc_list[i])

    return nuc_list


def add_missing_time_step(df, ref_time):
    """
    Add the missing time step to a Panda Data Frame.
    Parameters
    ----------
    df:  Pandas Data Frame
    ref_time:  list of the time step references (Coming from TimeStep metrics)
    """
    ref_time.rename(index=str, columns={'TimeStep':  'Time'}, inplace=True)

    if 'SimId' in ref_time.columns.values:
        ref_time.drop('SimId', 1, inplace=True)
    df = pd.merge(ref_time, df, how="outer")
    df.fillna(0, inplace=True)
    return df


def merge(df, base_col, add_df, add_col):
    """
    Merge some additionnal columns fram an additionnal Pandas Data Frame
    onother one and then remove the second base column (keeping SimID
    information).
    Parameters
    ----------
    df: Pandas Data Frame
    base_col: list of the base columns names
    add_df: Pandas Data Frame to add in the df one
    add_col: columns to be added
    """
    df = pd.merge(add_df[add_col], df, on=base_col)
    df.drop(base_col[1], 1)
    return df


def reduce(df, rdc_list):
    """
    Filter the df Pandas Data Frame according to the rdc_list (list of item
    in the corresponding columns).
    Parameters
    ----------
    df: Pandas Data Frame
    rdc_list: list of pair of string and string list.
    """
    for rdc in rdc_list:
        if len(rdc[1]) != 0:
            df = df[df[rdc[0]].isin(rdc[1])]
        else:
            wng_msg = "Empty list provided for " + rdc[0] + " key."
            warnings.warn(wng_msg, UserWarning)
    return df


def get_transaction_df(evaler_, send_list=[], rec_list=[], commod_list=[]):
    """
    Filter the Transaction Data Frame on specific sending facility and
    receving facility.

    Parameters
    ----------
    evaler_: evaler
    send_list: list of the sending facility
    rec_list: list of the receiving facility
    commod_list: list of the commodity exchanged
    """

    # initiate evaluation
    trans = evaler_.eval('Transactions')
    agents = evaler_.eval('AgentEntry')

    rec_agent = agents.rename(index=str, columns={'AgentId':  'ReceiverId'})
    if len(rec_list) != 0:
        rec_agent = rec_agent[rec_agent['Prototype'].isin(rec_list)]

    send_agent = agents.rename(index=str, columns={'AgentId':  'SenderId'})
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

        trans = reduce(trans, rdc_list)

        # Merge Sender to Transaction PDF
        base_col = ['SimId', 'SenderId']
        added_col = base_col + ['Prototype']
        trans = merge(trans, base_col, send_agent, added_col)
        trans = trans.rename(index=str, columns={'Prototype':  'SenderPrototype'})

        # Merge Receiver to Transaction PDF
        base_col = ['SimId', 'ReceiverId']
        added_col = base_col + ['Prototype']
        trans = merge(trans, base_col, rec_agent, added_col)
        trans = trans.rename(index=str, columns={'Prototype':  'ReceiverPrototype'})

    return trans


def get_transaction_nuc_df(evaler_, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Filter the Transaction Data Frame, which include nuclide composition, on specific sending facility and
    receving facility. Applying nuclides selection when required.

    Parameters
    ----------
    evaler_: evaler
    send_list: list of the sending facility
    rec_list: list of the receiving facility
    commod_list: list of the commodity exchanged
    nuc_list: list of nuclide to select.
    """

    compo = evaler_.eval('Materials')

    df = get_transaction_df(evaler_, send_list, rec_list, commod_list)

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)
        compo = reduce(compo, [['NucId', nuc_list]])

    base_col = ['SimId', 'ResourceId']
    added_col = base_col + ['NucId', 'Mass']
    df = merge(df, base_col, compo, added_col)

    return df


def get_transaction_activity_df(evaler_, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Return the transation df, with the activities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler_: evaler
    send_list: list of the sending facility
    rec_list: list of the receiving facility
    commod_list: list of the commodity exchanged
    nuc_list: list of nuclide to select.
    """

    df = get_transaction_df(evaler_, send_list, rec_list, commod_list)

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    compo = evaler_.eval('Activity')
    compo = reduce(compo, [['NucId', nuc_list]])

    base_col = ['SimId', 'ResourceId']
    added_col = base_col + ['NucId', 'Activity']
    df = merge(df, base_col, compo, added_col)

    return df


def get_transaction_decayheat_df(evaler_, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Return the transation df, with the decayheat. Applying nuclides selection when required.

    Parameters
    ----------
    evaler_: evaler
    send_list: list of the sending facility
    rec_list: list of the receiving facility
    commod_list: list of the commodity exchanged
    nuc_list: list of nuclide to select.
    """

    df = get_transaction_df(evaler_, send_list, rec_list, commod_list)

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    compo = evaler_.eval('DecayHeat')
    compo = reduce(compo, [['NucId', nuc_list]])

    base_col = ['SimId', 'ResourceId']
    added_col = base_col + ['NucId', 'DecayHeat']
    df = merge(df, base_col, compo, added_col)

    return df


def get_transaction_timeserie(evaler_, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Shape the reduced transation Data Frame into a simple time serie. Applying nuclides selection when required.

    Parameters
    ----------
    evaler_: evaler
    send_list: list of the sending facility
    rec_list: list of the receiving facility
    commod_list: list of the commodity exchanged
    nuc_list: list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    df = get_transaction_nuc_df(
        evaler_, send_list, rec_list, commod_list, nuc_list)

    group_end = ['ReceiverPrototype', 'SenderPrototype', 'Time']
    group_start = group_end + ['Mass']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    df = df[['Time', 'Mass']].groupby(['Time']).sum()
    df.reset_index(inplace=True)

    time = evaler_.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_transaction_activity_timeserie(evaler_, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Shape the reduced transation Data Frame into a simple time serie. Applying nuclides selection when required.

    Parameters
    ----------
    evaler_: evaler
    send_list: list of the sending facility
    rec_list: list of the receiving facility
    commod_list: list of the commodity exchanged
    nuc_list: list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    df = get_transaction_activity_df(evaler_, send_list, rec_list, commod_list,
                                     nuc_list)

    group_end = ['ReceiverPrototype', 'SenderPrototype', 'Time']
    group_start = group_end + ['Activity']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    df = df[['Time', 'Activity']].groupby(['Time']).sum()
    df.reset_index(inplace=True)

    time = evaler_.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_transaction_decayheat_timeserie(evaler_, send_list=[], rec_list=[], commod_list=[], nuc_list=[]):
    """
    Shape the reduced transation Data Frame into a simple time serie. Applying nuclides selection when required.

    Parameters
    ----------
    evaler_: evaler
    send_list: list of the sending facility
    rec_list: list of the receiving facility
    commod_list: list of the commodity exchanged
    nuc_list: list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    df = get_transaction_decayheat_df(evaler_, send_list, rec_list, commod_list,
                                      nuc_list)

    group_end = ['ReceiverPrototype', 'SenderPrototype', 'Time']
    group_start = group_end + ['DecayHeat']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    df = df[['Time', 'DecayHeat']].groupby(['Time']).sum()
    df.reset_index(inplace=True)

    time = evaler_.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_flow_graph(evaler_, send_list=[], rec_list=[], commod_list=[], nuc_list=[],
                   time=[-1, -1]):
    """
    Generate the dot graph of the transation between facilitiese. Applying times
    nuclides selection when required.

    Parameters
    ----------
    evaler_: evaler
    send_list: list of the sending facility
    rec_list: list of the receiving facility
    commod_list: list of the commodity exchanged
    nuc_list: list of nuclide to select.
    """
    tools.raise_no_graphviz('Unable to generate flow graph!', HAVE_GRAPHVIZ)

    df = get_transaction_nuc_df(
        evaler_, send_list, rec_list, commod_list, nuc_list)

    if time[0] != -1:
        df = df.loc[(df['Time'] > time[0])]
    if time[1] != -1:
        df = df.loc[(df['Time'] < time[1])]

    group_end = ['ReceiverPrototype', 'SenderPrototype']
    group_start = group_end + ['Mass']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    agents_list = evaler_.eval('AgentEntry')['Prototype'].tolist()

    dot = Digraph('G')

    for agent in agents_list:
        dot.node(agent)

    for index, row in df.iterrows():
        dot.edge(row['SenderPrototype'], row['ReceiverPrototype'],
                 label=str(row['Mass']))

    return dot


def get_inventory_df(evaler_, fac_list=[], nuc_list=[]):
    """
    Shape the reduced inventory Data Frame. Applying nuclides/facilities selection when required.

    Parameters
    ----------
    evaler_: evaler
    fac_list:  list of the facility
    nuc_list: list of nuclide to select.
    """

    # Get inventory table
    df = evaler_.eval('ExplicitInventory')
    agents = evaler_.eval('AgentEntry')

    rdc_list = []  # because we want to get rid of the nuclide asap
    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)
        rdc_list.append(['NucId', nuc_list])

    if len(fac_list) != 0:
        agents = agents[agents['Prototype'].isin(fac_list)]
        rdc_list.append(['AgentId', agents['AgentId'].tolist()])
    else:
        wng_msg = "no faciity provided"
        warnings.warn(wng_msg, UserWarning)
    df = reduce(df, rdc_list)

    base_col = ['SimId', 'AgentId']
    added_col = base_col + ['Prototype']
    df = merge(df, base_col, agents, added_col)

    return df


def get_inventory_timeserie(evaler_, fac_list=[], nuc_list=[]):
    """
    Shape the reduced inventory Data Frame into a simple time serie. Applying
    nuclides/facilities selection when required.

    Parameters
    ----------
    evaler_: evaler
    fac_list:  list of the facility
    nuc_list: list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)
    else:
        wng_msg = "no nuclide provided"
        warnings.warn(wng_msg, UserWarning)

    df = get_inventory_df(evaler_, fac_list, nuc_list)

    group_end = ['Time']
    group_start = group_end + ['Quantity']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    time = evaler_.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_inventory_activity_df(evaler_, fac_list=[], nuc_list=[]):
    """
    Get a simple time series of the activity of the inventory in the selcted
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler_: evaler
    fac_list:  list of the facility
    nuc_list: list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    df = get_inventory_df(evaler_, fac_list, nuc_list)
    for i, row in df.iterrows():
        val = 1000 * data.N_A * df.ix[i, 'Quantity'] * \
            data.decay_const(int(df.ix[i, 'NucId']))
        df.set_value(i, 'Activity', val)

    return df


def get_inventory_activity_timeserie(evaler_, fac_list=[], nuc_list=[]):
    """
    Get a simple time series of the decay heat of the inventory in the selcted
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler_: evaler
    fac_list:  list of the facility
    nuc_list: list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)
    else:
        wng_msg = "no nuclide provided"
        warnings.warn(wng_msg, UserWarning)

    df = get_inventory_activity_df(evaler_, fac_list, nuc_list)
    group_end = ['Time']
    group_start = group_end + ['Activity']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    time = evaler_.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_inventory_decayheat_df(evaler_, fac_list=[], nuc_list=[]):
    """
    Get a Inventory PDF including the decay heat of the inventory in the selected
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler_: evaler
    fac_list:  list of the facility
    nuc_list: list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)

    df = get_inventory_activity_df(evaler_, fac_list, nuc_list)
    for i, row in df.iterrows():
        val = data.MeV_per_MJ * \
            df.ix[i, 'Activity'] * data.q_val(int(df.ix[i, 'NucId']))
        df.set_value(i, 'DecayHeat', val)

    return df


def get_inventory_decayheat_timeserie(evaler_, fac_list=[], nuc_list=[]):
    """
    Get a simple time series of the decay heat of the inventory in the selcted
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler_: evaler
    fac_list: list of the facility
    nuc_list: list of nuclide to select.
    """

    if len(nuc_list) != 0:
        nuc_list = format_nuclist(nuc_list)
    else:
        wng_msg = "no nuclide provided"
        warnings.warn(wng_msg, UserWarning)

    df = get_inventory_decayheat_df(evaler_, fac_list, nuc_list)
    group_end = ['Time']
    group_start = group_end + ['DecayHeat']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    time = evaler_.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_power_timeserie(evaler_, fac_list=[]):
    """
    Shape the reduced Power Data Frame into a simple time serie. Applying
    facilities selection when required.

    Parameters
    ----------
    evaler_: evaler
    fac_list: list of the facility
    """

    # Get inventory table
    power = evaler_.eval('TimeSeriesPower')
    agents = evaler_.eval('AgentEntry')

    rdc_list = []  # because we want to get rid of the facility asap
    if len(fac_list) != 0:
        agents = agents[agents['Prototype'].isin(fac_list)]
        rdc_list.append(['AgentId', agents['AgentId'].tolist()])
    else:
        wng_msg = "no faciity provided"
        warnings.warn(wng_msg, UserWarning)
    power = reduce(power, rdc_list)

    base_col = ['SimId', 'AgentId']
    added_col = base_col + ['Prototype']
    power = merge(power, base_col, agents, added_col)

    group_end = ['Time']
    group_start = group_end + ['Value']
    df = power[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    time = evaler_.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_deployment_timeserie(evaler_, fac_list=[]):
    """
    Get a simple time series with deployment schedule of the selected facilities.

    Parameters
    ----------
    evaler_: evaler
    fac_list: list of the facility
    """

    # Get inventory table
    df = evaler_.eval('AgentEntry')

    rdc_list = []  # because we want to get rid of the facility asap
    if len(fac_list) != 0:
        df = df[df['Prototype'].isin(fac_list)]
        rdc_list.append(['AgentId', df['AgentId'].tolist()])
    else:
        wng_msg = "no faciity provided"
        warnings.warn(wng_msg, UserWarning)

    # Adding a constante column to easely sum the amount of facilities build per
    # time step
    df = df.assign(Value=lambda x:  1)
    group_end = ['EnterTime']
    group_start = group_end + ['Value']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)
    df.rename(index=str, columns={'EnterTime':  'Time'}, inplace=True)

    time = evaler_.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_retirement_timeserie(evaler_, fac_list=[]):
    """
    Get a simple time series with retirement schedule of the selected facilities.

    Parameters
    ----------
    evaler_: evaler
    fac_list:  list of the facility
    """

    # Get inventory table
    df = evaler_.eval('AgentEntry')
    df = df[df['Lifetime'] > 0]

    rdc_list = []  # because we want to get rid of the facility asap
    if len(fac_list) != 0:
        df = df[df['Prototype'].isin(fac_list)]
        rdc_list.append(['AgentId', df['AgentId'].tolist()])
    else:
        wng_msg = "no faciity provided"
        warnings.warn(wng_msg, UserWarning)

    # Adding a constante column to easely sum the amount of facilities build per
    # time stepi

    df = df.assign(Value=lambda x:  1)
    df['DecomTime'] = df['EnterTime'] + df['Lifetime']
    group_end = ['DecomTime']
    group_start = group_end + ['Value']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    df.rename(index=str, columns={'DecomTime':  'Time'}, inplace=True)
    time = evaler_.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df
