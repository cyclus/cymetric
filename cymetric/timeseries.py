import warning

import pandas as pd
import numpy as np

try:
    from pyne import data
    import pyne.enrichment as enr
    from pyne import nucname
    HAVE_PYNE = True
except ImportError:
    HAVE_PYNE = False


import cymetric as cym
from cymetric import tools

from cymetric.filter import get_transaction_df
from cymetric.filter import get_transaction_nuc_df
from cymetric.filter import get_transaction_activity_df
from cymetric.filter import get_transaction_decayheat_df
from cymetric.filter import get_inventory_df
from cymetric.filter import get_inventory_activity_df
from cymetric.filter import get_inventory_decayheat_df


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
