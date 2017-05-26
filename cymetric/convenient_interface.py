import warnings

import pandas as pd
import numpy as np

try:
    from graphviz import Digraph
    HAVE_GRAPHVIZ = True
except ImportError:
    HAVE_GRAPHVIZ = False

try:
    from pyne import data
    import pyne.enrichment as enr
    from pyne import nucname
    HAVE_PYNE = True
except ImportError:
    HAVE_PYNE = False


import cymetric as cym
from cymetric import tools

def format_nuc(nucs):
    """
    format the nuclide  provided by the users into a standard format:
    ZZAASSSS.

    Parameters
    ----------
    nucs :  of nuclides
    """
    tools.raise_no_pyne('Unable to format nuclide !', HAVE_PYNE)

    return [nucname.id(nuc) for nuc in nucs]


def add_missing_time_step(df, ref_time):
    """
    Add the missing time step to a Panda Data Frame.

    Parameters
    ----------
    df : Pandas Data Frame
    ref_time :  of the time step references (Coming from TimeStep metrics)
    """
    ref_time.rename(index=str, columns={'TimeStep': 'Time'}, inplace=True)

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
    base_col:  of the base columns names
    add_df: Pandas Data Frame to add in the df one
    add_col: columns to be added
    """
    df = pd.merge(add_df[add_col], df, on=base_col)
    df.drop(base_col[1], 1)
    return df


def reduce(df, rdc_):
    """
    Filter the df Pandas Data Frame according to the rdc_ (list of item
    in the corresponding columns).

    Parameters
    ----------
    df: Pandas Data Frame
    rdc_: list of pair of string and string list.
    """
    for rdc in rdc_:
        if len(rdc[1]) != 0:
            df = df[df[rdc[0]].isin(rdc[1])]
        else:
            wng_msg = "Empty  provided for " + rdc[0] + " key."
            warnings.warn(wng_msg, UserWarning)
    return df


def get_transaction_df(evaler, senders=(), receivers=(), commodities=()):
    """
    Filter the Transaction Data Frame on specific sending facility and
    receving facility.

    Parameters
    ----------
    evaler : evaler
    senders :  of the sending facility
    receivers :  of the receiving facility
    commodities :  of the commodity exchanged
    """

    # initiate evaluation
    trans = evaler.eval('Transactions')
    agents = evaler.eval('AgentEntry')

    rec_agent = agents.rename(index=str, columns={'AgentId': 'ReceiverId'})
    if len(receivers) != 0:
        rec_agent = rec_agent[rec_agent['Prototype'].isin(receivers)]

    send_agent = agents.rename(index=str, columns={'AgentId': 'SenderId'})
    if len(senders) != 0:
        send_agent = send_agent[send_agent['Prototype'].isin(senders)]

    # check if sender and receiver exist
    if rec_agent.empty or send_agent.empty:
        return None
    
    # Clean Transation PDF
    rdc_ = []
    rdc_.append(['ReceiverId', rec_agent['ReceiverId'].tolist()])
    rdc_.append(['SenderId', send_agent['SenderId'].tolist()])
    if len(commodities) != 0:
        rdc_.append(['Commodity', commodities])

    trans = reduce(trans, rdc_)

    # Merge Sender to Transaction PDF
    base_col = ['SimId', 'SenderId']
    added_col = base_col + ['Prototype']
    trans = merge(trans, base_col, send_agent, added_col)
    trans = trans.rename(index=str, columns={
                         'Prototype': 'SenderPrototype'})

    # Merge Receiver to Transaction PDF
    base_col = ['SimId', 'ReceiverId']
    added_col = base_col + ['Prototype']
    trans = merge(trans, base_col, rec_agent, added_col)
    trans = trans.rename(index=str, columns={
                         'Prototype': 'ReceiverPrototype'})

    return trans


def get_transaction_nuc_df(evaler, senders=(), receivers=(), commodities=(), nucs=()):
    """
    Filter the Transaction Data Frame, which include nuclide composition, on specific sending facility and
    receving facility. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    senders :  of the sending facility
    receivers :  of the receiving facility
    commodities :  of the commodity exchanged
    nucs :  of nuclide to select.
    """

    compo = evaler.eval('Materials')

    df = get_transaction_df(evaler, senders, receivers, commodities)

    if len(nucs) != 0:
        nucs = format_nuc(nucs)
        compo = reduce(compo, [['NucId', nucs]])

    base_col = ['SimId', 'ResourceId']
    added_col = base_col + ['NucId', 'Mass']
    df = merge(df, base_col, compo, added_col)

    return df


def get_transaction_activity_df(evaler, senders=(), receivers=(), commodities=(), nucs=()):
    """
    Return the transation df, with the activities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    senders :  of the sending facility
    receivers :  of the receiving facility
    commodities :  of the commodity exchanged
    nucs :  of nuclide to select.
    """

    df = get_transaction_df(evaler, senders, receivers, commodities)

    if len(nucs) != 0:
        nucs = format_nuc(nucs)

    compo = evaler.eval('Activity')
    compo = reduce(compo, [['NucId', nucs]])

    base_col = ['SimId', 'ResourceId']
    added_col = base_col + ['NucId', 'Activity']
    df = merge(df, base_col, compo, added_col)

    return df


def get_transaction_decayheat_df(evaler, senders=(), receivers=(), commodities=(), nucs=()):
    """
    Return the transation df, with the decayheat. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    senders :  of the sending facility
    receivers :  of the receiving facility
    commodities :  of the commodity exchanged
    nucs :  of nuclide to select.
    """

    df = get_transaction_df(evaler, senders, receivers, commodities)

    if len(nucs) != 0:
        nucs = format_nuc(nucs)

    compo = evaler.eval('DecayHeat')
    compo = reduce(compo, [['NucId', nucs]])

    base_col = ['SimId', 'ResourceId']
    added_col = base_col + ['NucId', 'DecayHeat']
    df = merge(df, base_col, compo, added_col)

    return df


def get_transaction_timeseries(evaler, senders=(), receivers=(), commodities=(), nucs=()):
    """
    Shape the reduced transation Data Frame into a simple time serie. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    senders :  of the sending facility
    receivers :  of the receiving facility
    commodities :  of the commodity exchanged
    nucs :  of nuclide to select.
    """

    if len(nucs) != 0:
        nucs = format_nuc(nucs)

    df = get_transaction_nuc_df(
        evaler, senders, receivers, commodities, nucs)

    group_end = ['ReceiverPrototype', 'SenderPrototype', 'Time']
    group_start = group_end + ['Mass']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    df = df[['Time', 'Mass']].groupby(['Time']).sum()
    df.reset_index(inplace=True)

    time = evaler.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_transaction_activity_timeseries(evaler, senders=(), receivers=(), commodities=(), nucs=()):
    """
    Shape the reduced transation Data Frame into a simple time serie. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    senders :  of the sending facility
    receivers :  of the receiving facility
    commodities :  of the commodity exchanged
    nucs :  of nuclide to select.
    """

    if len(nucs) != 0:
        nucs = format_nuc(nucs)

    df = get_transaction_activity_df(evaler, senders, receivers, commodities,
                                     nucs)

    group_end = ['ReceiverPrototype', 'SenderPrototype', 'Time']
    group_start = group_end + ['Activity']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    df = df[['Time', 'Activity']].groupby(['Time']).sum()
    df.reset_index(inplace=True)

    time = evaler.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_transaction_decayheat_timeseries(evaler, senders=(), receivers=(), commodities=(), nucs=()):
    """
    Shape the reduced transation Data Frame into a simple time serie. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    senders :  of the sending facility
    receivers :  of the receiving facility
    commodities :  of the commodity exchanged
    nucs :  of nuclide to select.
    """

    if len(nucs) != 0:
        nucs = format_nuc(nucs)

    df = get_transaction_decayheat_df(evaler, senders, receivers, commodities,
                                      nucs)

    group_end = ['ReceiverPrototype', 'SenderPrototype', 'Time']
    group_start = group_end + ['DecayHeat']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    df = df[['Time', 'DecayHeat']].groupby(['Time']).sum()
    df.reset_index(inplace=True)

    time = evaler.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_flow_graph(evaler, senders=(), receivers=(), commodities=(), nucs=(),
                   time=[-1, -1]):
    """
    Generate the dot graph of the transation between facilitiese. Applying times
    nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    senders :  of the sending facility
    receivers :  of the receiving facility
    commodities :  of the commodity exchanged
    nucs :  of nuclide to select.
    """
    tools.raise_no_graphviz('Unable to generate flow graph!', HAVE_GRAPHVIZ)

    df = get_transaction_nuc_df(
        evaler, senders, receivers, commodities, nucs)

    if time[0] != -1:
        df = df.loc[(df['Time'] > time[0])]
    if time[1] != -1:
        df = df.loc[(df['Time'] < time[1])]

    group_end = ['ReceiverPrototype', 'SenderPrototype']
    group_start = group_end + ['Mass']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    agents_ = evaler.eval('AgentEntry')['Prototype'].tolist()

    dot = Digraph('G')

    for agent in agents_:
        dot.node(agent)

    for index, row in df.iterrows():
        dot.edge(row['SenderPrototype'], row['ReceiverPrototype'],
                 label=str(row['Mass']))

    return dot


def get_inventory_df(evaler, facilities=(), nucs=()):
    """
    Shape the reduced inventory Data Frame. Applying nuclides/facilities selection when required.

    Parameters
    ----------
    evaler : evaler
    facilities :  of the facility
    nucs :  of nuclide to select.
    """

    # Get inventory table
    df = evaler.eval('ExplicitInventory')
    agents = evaler.eval('AgentEntry')

    rdc_ = []  # because we want to get rid of the nuclide asap
    if len(nucs) != 0:
        nucs = format_nuc(nucs)
        rdc_.append(['NucId', nucs])

    if len(facilities) != 0:
        agents = agents[agents['Prototype'].isin(facilities)]
        rdc_.append(['AgentId', agents['AgentId'].tolist()])
    else:
        wng_msg = "no faciity provided"
        warnings.warn(wng_msg, UserWarning)
    df = reduce(df, rdc_)

    base_col = ['SimId', 'AgentId']
    added_col = base_col + ['Prototype']
    df = merge(df, base_col, agents, added_col)

    return df


def get_inventory_timeseries(evaler, facilities=(), nucs=()):
    """
    Shape the reduced inventory Data Frame into a simple time serie. Applying
    nuclides/facilities selection when required.

    Parameters
    ----------
    evaler : evaler
    facilities :  of the facility
    nucs :  of nuclide to select.
    """

    if len(nucs) != 0:
        nucs = format_nuc(nucs)
    else:
        wng_msg = "no nuclide provided"
        warnings.warn(wng_msg, UserWarning)

    df = get_inventory_df(evaler, facilities, nucs)

    group_end = ['Time']
    group_start = group_end + ['Quantity']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    time = evaler.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_inventory_activity_df(evaler, facilities=(), nucs=()):
    """
    Get a simple time series of the activity of the inventory in the selcted
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    facilities :  of the facility
    nucs :  of nuclide to select.
    """

    if len(nucs) != 0:
        nucs = format_nuc(nucs)

    df = get_inventory_df(evaler, facilities, nucs)
    for i, row in df.iterrows():
        val = 1000 * data.N_A * row['Quantity'] * \
            data.decay_const(int(row['NucId']))
        df.set_value(i, 'Activity', val)

    return df


def get_inventory_activity_timeseries(evaler, facilities=(), nucs=()):
    """
    Get a simple time series of the decay heat of the inventory in the selcted
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    facilities :  of the facility
    nucs :  of nuclide to select.
    """

    if len(nucs) != 0:
        nucs = format_nuc(nucs)
    else:
        wng_msg = "no nuclide provided"
        warnings.warn(wng_msg, UserWarning)

    df = get_inventory_activity_df(evaler, facilities, nucs)
    group_end = ['Time']
    group_start = group_end + ['Activity']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    time = evaler.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_inventory_decayheat_df(evaler, facilities=(), nucs=()):
    """
    Get a Inventory PDF including the decay heat of the inventory in the selected
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    facilities :  of the facility
    nucs :  of nuclide to select.
    """

    if len(nucs) != 0:
        nucs = format_nuc(nucs)

    df = get_inventory_activity_df(evaler, facilities, nucs)
    for i, row in df.iterrows():
        val = data.MeV_per_MJ * \
            row['Activity'] * data.q_val(int(row['NucId']))
        df.set_value(i, 'DecayHeat', val)

    return df


def get_inventory_decayheat_timeseries(evaler, facilities=(), nucs=()):
    """
    Get a simple time series of the decay heat of the inventory in the selcted
    facilities. Applying nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    facilities :  of the facility
    nucs :  of nuclide to select.
    """

    if len(nucs) != 0:
        nucs = format_nuc(nucs)
    else:
        wng_msg = "no nuclide provided"
        warnings.warn(wng_msg, UserWarning)

    df = get_inventory_decayheat_df(evaler, facilities, nucs)
    group_end = ['Time']
    group_start = group_end + ['DecayHeat']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    time = evaler.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_power_timeseries(evaler, facilities=()):
    """
    Shape the reduced Power Data Frame into a simple time serie. Applying
    facilities selection when required.

    Parameters
    ----------
    evaler : evaler
    facilities :  of the facility
    """

    # Get inventory table
    power = evaler.eval('TimeSeriesPower')
    agents = evaler.eval('AgentEntry')

    rdc_ = []  # because we want to get rid of the facility asap
    if len(facilities) != 0:
        agents = agents[agents['Prototype'].isin(facilities)]
        rdc_.append(['AgentId', agents['AgentId'].tolist()])
    else:
        wng_msg = "no faciity provided"
        warnings.warn(wng_msg, UserWarning)
    power = reduce(power, rdc_)

    base_col = ['SimId', 'AgentId']
    added_col = base_col + ['Prototype']
    power = merge(power, base_col, agents, added_col)

    group_end = ['Time']
    group_start = group_end + ['Value']
    df = power[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    time = evaler.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_deployment_timeseries(evaler, facilities=()):
    """
    Get a simple time series with deployment schedule of the selected facilities.

    Parameters
    ----------
    evaler : evaler
    facilities :  of the facility
    """

    # Get inventory table
    df = evaler.eval('AgentEntry')

    rdc_ = []  # because we want to get rid of the facility asap
    if len(facilities) != 0:
        df = df[df['Prototype'].isin(facilities)]
        rdc_.append(['AgentId', df['AgentId'].tolist()])
    else:
        wng_msg = "no faciity provided"
        warnings.warn(wng_msg, UserWarning)

    # Adding a constant column to easily sum the amount of facilities build per
    # time step
    df = df.assign(Value=lambda x: 1)
    group_end = ['EnterTime']
    group_start = group_end + ['Value']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)
    df.rename(index=str, columns={'EnterTime': 'Time'}, inplace=True)

    time = evaler.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df


def get_retirement_timeseries(evaler, facilities=()):
    """
    Get a simple time series with retirement schedule of the selected facilities.

    Parameters
    ----------
    evaler : evaler
    facilities :  of the facility
    """

    # Get inventory table
    df = evaler.eval('AgentEntry')
    df = df[df['Lifetime'] > 0]

    rdc_ = []  # because we want to get rid of the facility asap
    if len(facilities) != 0:
        df = df[df['Prototype'].isin(facilities)]
        rdc_.append(['AgentId', df['AgentId'].tolist()])
    else:
        wng_msg = "no faciity provided"
        warnings.warn(wng_msg, UserWarning)

    # Adding a constant column to easily sum the amount of facilities build per
    # time step i

    df = df.assign(Value=lambda x: 1)
    df['DecomTime'] = df['EnterTime'] + df['Lifetime']
    group_end = ['DecomTime']
    group_start = group_end + ['Value']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    df.rename(index=str, columns={'DecomTime': 'Time'}, inplace=True)
    time = evaler.eval('TimeList')
    df = add_missing_time_step(df, time)
    return df
