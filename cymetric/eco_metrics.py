# economic metrics for nuclear power plants #####

from __future__ import print_function, unicode_literals

import inspect

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import math
import os
import warnings

from cyclus import typesystem as ts


try:
    from cymetric.metrics import metric
    from cymetric.tools import dbopen, merge
    from cymetric.evaluator import Evaluator
    from cymetric import eco_tools
    from cymetric.evaluator import register_metric
except ImportError:
    # some wacky CI paths prevent absolute importing, try relative
    from .metrics import metric
    from .evaluator import register_metric
    from . import eco_tools
    from .tools import dbopen, merge
    from .evaluator import Evaluator


# This xml file has to be built in the same direction as the sqlite output
# database. It contains the economic data needed to calculate the EconomicInfo
# metric
xml_inputs = 'parameters.xml'


# The actual metrics ##
eco_data = None


_ccdeps = ['TimeSeriesPower', 'AgentEntry', 'Info']

_ccschema = [
    ('SimId', ts.UUID),
    ('AgentId', ts.INT),
    ('Time', ts.INT),
    ('Payment', ts.DOUBLE)]


@metric(name='CapitalCost', depends=_ccdeps, schema=_ccschema)
def capital_cost(dfPower, dfEntry, dfInfo):
    """The CapitalCost metric gives the cash flows at each time step
    corresponding to the reactors construction costs.
    """
    if eco_data is None:
        print("Warning! No economical data, please load one!")
        return None

    simDuration = dfInfo['Duration'].iloc[0]
    rtn = pd.DataFrame()
    for index, power_row in dfPower.iterrows():
        agent_id = power_row['AgentId']
        filiation = eco_tools.get_filiation_per_id(agent_id, dfEntry)
        agent_eco_prop = eco_data.get_prototype_eco(filiation)

        deviation = float(agent_eco_prop["capital"]["deviation"])
        deviation *= np.random.randn(1)
        deviation = int(np.around(deviation))

        beforePeak = int(agent_eco_prop["capital"]["beforePeak"]) + deviation
        afterPeak = int(agent_eco_prop["capital"]["afterPeak"])
        cashFlowShape = eco_tools.capital_shape(beforePeak, afterPeak)

        constructionDuration = int(
            agent_eco_prop["capital"]['constructionDuration']) + deviation

        powerCapacity = power_row['Value']
        overnightCost = float(agent_eco_prop["capital"]['overnight_cost'])
        cashFlow = np.around(cashFlowShape
                             * overnightCost
                             * powerCapacity, 4)

        discountRate = agent_eco_prop["finance"]['discount_rate']
        cashFlow *= ((1 + discountRate)
                     ** math.ceil((beforePeak + afterPeak) / 12) -
                     1) / (discountRate
                           * math.ceil((beforePeak + afterPeak) / 12))
        entry_time = dfEntry[
            dfEntry['AgentId'] == agent_id].iloc[0]["EnterTime"]
        TimeSerie = list(range(beforePeak + afterPeak + 1))
        TimeSerie += entry_time - constructionDuration
        tmp = pd.DataFrame(data={'AgentId': agent_id,
                                 'Time': TimeSerie,
                                 'Payment': cashFlow},
                           columns=['AgentId', 'Time', 'Payment'])
        rtn = pd.concat([rtn, tmp], ignore_index=False, sort=False)

    rtn['SimId'] = dfPower['SimId'].iloc[0]
    rtn = rtn[rtn['Time'].apply(lambda x: x >= 0 and x < simDuration)]
    rtn = rtn.reset_index()
    return rtn[['SimId', 'AgentId', 'Time', 'Payment']]


del _ccdeps, _ccschema


_fcdeps = ['Resources', 'Transactions', 'AgentEntry']

_fcschema = [
    ('SimId', ts.UUID),
    ('TransactionId', ts.INT),
    ('AgentId', ts.INT),
    ('Commodity', ts.STRING),
    ('Payment', ts.DOUBLE),
    ('Time', ts.INT)]


@metric(name='FuelCost', depends=_fcdeps, schema=_fcschema)
def fuel_cost(dfResources, dfTransactions, dfEntry):
    """The FuelCost metric gives the cash flows at each time step corresponding
    to the reactors fuel costs. It also contains the waste fee.
    """

# Unsure if it is about Sender or Receiver implementation here and test are not
# in agreement, taking receiver (using implementation as ref)
    rtn = dfTransactions.rename(columns={'ReceiverId': 'AgentId'})

    # Get eco data
    agents_eco_prop = eco_data.get_prototypes_eco()

    # Add quantity to Transaction
    base_col = ['SimId', 'ResourceId']
    added_col = base_col + ['Quantity']
    rtn = merge(rtn, base_col, dfResources, added_col)

    # Adding prototype info
    base_col = ['SimId', 'AgentId']
    added_col = base_col + ['Prototype']
    rtn = merge(rtn, base_col, dfEntry, added_col)

    # Merge Eco with Transaction per ReceiverId and commodity
    base_col = ['Prototype', 'Commodity']
    added_col = base_col + ['supply_cost',
                            'waste_fee', 'fuel_dev']
    rtn = merge(rtn, base_col, agents_eco_prop, added_col)

    for index, row in rtn.iterrows():
        rtn.at[index, 'fuel_dev'] *= np.random.randn(1)
    rtn['Payment'] = rtn['Quantity'] * \
        (rtn['fuel_dev'] + rtn['supply_cost'] + rtn['waste_fee'])
    rtn = rtn[['SimId',
               'TransactionId',
               'AgentId',
               'Commodity',
               'Payment',
               'Time']]
    return rtn


del _fcdeps, _fcschema


_dcdeps = ['TimeSeriesPower', 'AgentEntry', 'Info']

_dcschema = [
    ('SimId', ts.UUID),
    ('AgentId', ts.INT),
    ('Payment', ts.DOUBLE),
    ('Time', ts.INT)]


@metric(name='DecommissioningCost', depends=_dcdeps, schema=_dcschema)
def decommissioning_cost(dfPower, dfEntry, dfInfo):
    """The Decommissioning cost metric gives the cash flows at each time step
    corresponding to the reactors decommissioning.
    """
    out_col = ['SimId', 'AgentId', 'Payment', 'Time']
    # Get eco data
    dfEcoInfo = eco_data.get_prototypes_eco()

    dfEntry = dfEntry[dfEntry['Lifetime'].apply(lambda x: x > 0)]

    # if empty do nothing
    if len(dfEntry) == 0:
        return pd.DataFrame(columns=out_col)

    reactorsId = dfEntry[dfEntry['Spec'].apply(
        lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    rtn = pd.DataFrame()
    for id in reactorsId:
        proto = dfEntry[dfEntry['AgentId'] == id].at[0, 'Prototype']
        tmp = dfEcoInfo[dfEcoInfo['Prototype'] == proto].reset_index()
        if (len(tmp) == 1):
            duration = int(tmp.at[0, 'decom_duration'])
            overnightCost = tmp.at[0, 'decom_overnight_cost']
            cashFlowShape = eco_tools.capital_shape(
                duration - duration // 2, duration // 2)
            powerCapacity = dfPower[dfPower['AgentId']
                                    == id].reset_index().at[0, 'Value']
            cashFlow = cashFlowShape * powerCapacity * overnightCost
            entryTime = dfEntry[dfEntry.AgentId == id]['EnterTime'][0]
            lifetime = dfEntry[dfEntry.AgentId == id]['Lifetime'][0]
            time_pdf = pd.DataFrame({'AgentId': id,
                                     'Time': list(range(lifetime + entryTime,
                                                        lifetime + entryTime
                                                        + duration + 1)),
                                     'Payment': cashFlow})
            rtn = pd.concat([rtn, time_pdf], ignore_index=True)
    rtn['SimId'] = dfPower['SimId'][0]
    return rtn[out_col]


del _dcdeps, _dcschema


_omdeps = ['TimeSeriesPower', 'AgentEntry']

_omschema = [('SimId', ts.UUID),
             ('AgentId', ts.INT),
             ('Time', ts.INT),
             ('Payment', ts.DOUBLE)]


@metric(name='OperationMaintenance', depends=_omdeps, schema=_omschema)
def operation_maintenance(dfPower, dfEntry):
    """The OperationMaintenance metric gives the cash flows at each time step
    corresponding to the reactor operations and maintenance costs.
    """
    exp = ['SimId', 'AgentId', 'Time', 'Payment']
    power = ['SimId', 'AgentId', 'Time', 'Value']
    ecoInfo = ['fixed', 'variable', 'operation_dev']
    # Get eco data
    dfEcoInfo = eco_data.get_prototypes_eco()

    # Adding prototype info
    base_col = ['Prototype']
    added_col = base_col + ecoInfo
    rtn = pd.merge(dfEntry, dfEcoInfo[added_col], on=base_col)

    for index, row in rtn.iterrows():
        rtn.at[index, 'operation_dev'] *= np.random.randn(1)
    rtn['fixed'] += rtn['operation_dev']
    rtn['variable'] += rtn['operation_dev']

    base_col = ['AgentId']
    added_col = base_col + ['Time', 'Value']
    rtn = pd.merge(dfPower[added_col], rtn, on=base_col)
    rtn['Value'] *= 8760 / 12
    rtn['Payment'] = (rtn['Value'] * rtn['variable'] +
                      max(rtn['Value']) * rtn['fixed'])
    return rtn[['SimId', 'AgentId', 'Time', 'Payment']]


del _omdeps, _omschema


"""The functions below aim at calculating more complex metrics than the simple
cash flows corresponding to the construction, fuel, O&M and decommissioning
costs. The metrics can be calculated at an agent, institution, region or
simulation level.
"""
#######################################
# Metrics derived from the cash flows #
#######################################


_omdeps = ['CapitalCost',
           'DecommissioningCost',
           'OperationMaintenance',
           'FuelCost']

_omschema = [('SimId', ts.UUID),
             ('Time', ts.INT),
             ('AgentId', ts.INT),
             ('Fuel', ts.DOUBLE),
             ('OperationMaintenance', ts.DOUBLE),
             ('Decommission', ts.DOUBLE),
             ('Capital', ts.DOUBLE)]


@metric(name='MonthlyCosts', depends=_omdeps, schema=_omschema)
def montly_costs(dfCapitalCost, dfDecom, dfOM, dfFuelCost):

    base_col = ['SimId', 'AgentId', 'Time']
    costs = pd.DataFrame(columns=base_col)

    costs_pairs = [(dfCapitalCost, "Capital"),
                   (dfDecom, "Decommission"),
                   (dfOM, "OperationMaintenance"),
                   (dfFuelCost, "Fuel")]
    for pair in costs_pairs:
        df_tmp = pair[0]
        df_tmp = df_tmp.groupby(base_col).sum().reset_index()
        df_tmp = df_tmp.rename(columns={'Payment': pair[1]})
        if (len(df_tmp) != 0):
            costs = pd.merge(df_tmp, costs,
                             how='outer', on=base_col)
        else:
            costs[pair[1]] = 0

    costs = costs.fillna(0)
    return costs.drop(['TransactionId'], axis=1)


del _omdeps, _omschema


def annual_costs(evaler, agents=(), agentsId=(), capital=True):
    """Input : sqlite output database and reactor's AgentId. It is possible not
    to take into account the construction costs (capital=False) if the reactor
    is supposed to have been built before the beginning of the simulation.
    Output : total reactor costs per year over its lifetime.
    """

    costs = evaler.eval('MonthlyCosts')

    if len(agents) != 0:
        dfAgents = evaler.eval('AgentEntry')
        agentsId += dfAgents[dfAgents['Prototype'].isin(
            agents)]["AgentId"].tolist()
    if len(agentsId) != 0:
        costs = costs[costs['AgentId'].isin(agentsId)]

    dfInfo = evaler.eval('Info')
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']

    costs['Year'] = (costs['Time'] + initialMonth - 1) // 12 + initialYear
    costs = costs.groupby(['Year', "AgentId"]).sum().reset_index()
    costs.drop(['Time'], axis=1, inplace=True)

    if not capital:
        del costs['Capital']

    return costs


def child_annual_costs(evaler, agents=(), agentsId=(), capital=True):
    """Input : evaler database and institution's AgentId. It is
    possible not to take into account the construction costs (capital=False)
    if the reactors are supposed to have been built before the beginning of
    the simulation. It is also possible to truncate the simulation results and
    only have access to cash flows occurring between the two dates (begin and
    end) specified in 'parameters.xml'. The truncation allows to let reactors
    decommission after the end of the simulation and thus print take into account
    cash flows that occur after the end of the simulation for example to
    calculate the LCOE.
    Output : total reactor costs per year over its lifetime at the institution
    level.
    """
    dfEntry = evaler.eval('AgentEntry')
    childId = eco_tools.get_child_id(agentsId, dfEntry)
    agentsId += eco_tools.get_child_id(agentsId, dfEntry)
    return annual_costs(evaler, agentsId=childId, capital=capital)


def all_annual_costs(evaler, agents=(), agentsId=(), capital=True):
    """Input : evaler database and region's AgentId. It is possible not
    to take into account the construction costs (capital=False) if the reactors
    are supposed to have been built before the beginning of the simulation. It
    is also possible to truncate the simulation results and only have access
    to cash flows occurring between the two dates (begin and end) specified in
    'parameters.xml'. The truncation allows to let reactors decommission after
    the end of the simulation and thus to take into account cash flows that
    occur after the end of the simulation for example to calculate the LCOE.

    Output : total reactor costs per year over its lifetime at the region
    level.
    """
    dfEntry = evaler.eval('AgentEntry')
    agentsId += eco_tools.get_child_id(agentsId, dfEntry)
    return annual_costs(evaler, agentsId=agentsId, capital=capital)


def actualize_costs(df, columns,  dfEntry, time_col='Time', time_factor=12., t_0=0):
    df_actualised = df.copy()

    for index, row in df_actualised.iterrows():
        agent_eco = eco_data.get_prototype_eco(
            eco_tools.get_filiation_per_id(row['AgentId'], dfEntry))
        discount_rate = agent_eco['finance']['discount_rate']
        t = row[time_col] / time_factor
        actualization = eco_tools.actualize(t-t_0, discount_rate)

        for col in columns:
            df_actualised.at[index, col] *= actualization
    return df_actualised


_omdeps = ['MonthlyCosts', 'AgentEntry']

_omschema = [('SimId', ts.UUID),
             ('AgentId', ts.INT),
             ('Fuel', ts.DOUBLE),
             ('OperationMaintenance', ts.DOUBLE),
             ('Decommission', ts.DOUBLE),
             ('Capital', ts.DOUBLE)]


@ metric(name='ActualizedMonthlyCosts', depends=_omdeps, schema=_omschema)
def actualised_montly_costs(dfMontlyCost, dfEntry):

    df_actualised = dfMontlyCost.copy()

    col_to_actualize = ['Fuel',
                        'OperationMaintenance',
                        'Decommission',
                        'Capital']
    return actualize_costs(df_actualised, col_to_actualize, dfEntry, time_col='Time', time_factor=12.)


del _omdeps, _omschema


def actualized_annual_costs(evaler, agents=(), agentsId=(), capital=True):
    """Same as annual_cost except all values are actualized to the begin date
    of the SIMULATION
    """

    costs = evaler.eval('ActualizedMonthlyCosts').copy()

    if len(agents) != 0:
        dfAgents = evaler.eval('AgentEntry').copy()
        agentsId += dfAgents[dfAgents['Prototype'].isin(
            agents)]["AgentId"].tolist()
    if len(agentsId) != 0:
        costs = costs[costs['AgentId'].isin(agentsId)]

    dfInfo = evaler.eval('Info')
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']

    costs['Year'] = (costs['Time'] + initialMonth - 1) // 12 + initialYear
    costs = costs.groupby(['Year', "AgentId"]).sum().reset_index()
    costs.drop(['Time'], axis=1, inplace=True)

    if not capital:
        del costs['Capital']

    return costs


def child_actualized_annual_costs(evaler, agents=(), agentsId=(), capital=True):
    """Same as annual_cost except all values are actualized to the begin date
    of the SIMULATION
    """
    dfEntry = evaler.eval('AgentEntry')
    childId = eco_tools.get_child_id(agentsId, dfEntry)
    return actualized_annual_costs(evaler, agentsId=childId, capital=capital)


def all_actualized_annual_costs(evaler, agents=[], agentsId=[], capital=True):
    """Same as annual_cost except all values are actualized to the begin date
    of the SIMULATION
    """
    dfEntry = evaler.eval('AgentEntry')
    agentsId += eco_tools.get_child_id(agentsId, dfEntry)
    return actualized_annual_costs(evaler, agentsId=agentsId, capital=capital)


def actualized_annual_cost(evaler, agents=(), agentsId=(), capital=True):
    """Same as annual_cost except all values are actualized to the begin date
    of the SIMULATION
    """
    cost = actualized_annual_costs(evaler, agents, agentsId, capital)
    cost['Cost'] = cost['Fuel'] + cost['OperationMaintenance'] + \
        cost['Capital'] + cost['Decommission']
    return cost[['Year', 'AgentId', 'Cost']]


def child_actualized_annual_cost(evaler, agents=(), agentsId=(), capital=True):
    """Same as annual_cost except all values are actualized to the begin date
    of the SIMULATION
    """
    cost = child_actualized_annual_costs(evaler, agents, agentsId, capital)
    cost['Cost'] = cost['Fuel'] + cost['OperationMaintenance'] + \
        cost['Capital'] + cost['Decommission']
    return cost[['Year', 'AgentId', 'Cost']]


def all_actualized_annual_cost(evaler, agents=[], agentsId=[], capital=True):
    """Same as annual_cost except all values are actualized to the begin date
    of the SIMULATION
    """
    cost = all_actualized_annual_costs(evaler, agents, agentsId, capital)
    cost['Cost'] = cost['Fuel'] + cost['OperationMaintenance'] + \
        cost['Capital'] + cost['Decommission']
    return cost[['Year', 'AgentId', 'Cost']]


def average_cost(evaler, reactorId, capital=True):
    """Input : evaler, reactor's AgentId
    Output : value (in $/MWh) corresponding to the total costs (sum of annual
    costs) divided by the total power generated.
    """

    powerGenerated = power_generated(evaler, [reactorId])['Energy'].sum()
    costs = actualized_annual_costs(evaler,
                                    agentsId=[reactorId],
                                    capital=capital)
    # removing Year and AgentId column (for the sum)
    costs.drop(['Year', 'AgentId'], axis=1, inplace=True)

    return costs.sum().sum() / powerGenerated


def power_generated(evaler, agentsId=[]):
    """Input : cymetric evaler and reactor agent id
    Output : Electricity generated in MWh every year
    """
    dfPower = evaler.eval('AnnualElectricityGeneratedByAgent').copy()
    initialYear = evaler.eval('Info').loc[0, 'InitialYear']

    # Convert to absolute time
    dfPower['Year'] = dfPower['Year'] + initialYear
    # Convert Power to MWh
    dfPower['Energy'] *= 365.25 * 24

    # Return table filtered by AgentsId
    return dfPower[dfPower['AgentId'].isin(agentsId)]


def child_power_generated(evaler, agentsId=[]):
    """Input : cymetric evaler and reactor agent id
    Output : Electricity generated in MWh every year
    """
    dfEntry = evaler.eval('AgentEntry')
    childId = eco_tools.get_child_id(agentsId, dfEntry)
    return power_generated(evaler, agentsId=childId)


def all_power_generated(evaler, agentsId=[]):
    """Input : cymetric evaler and reactor agent id
    Output : Electricity generated in MWh every year
    """
    dfEntry = evaler.eval('AgentEntry')
    agentsId += eco_tools.get_child_id(agentsId, dfEntry)
    return power_generated(evaler, agentsId=agentsId)


def benefit(evaler, agentsId=[]):
    """Input : evaler database and reactor agent id
    Output : cumulative sum of actualized income and expense
             (= - expenditures + income)
    """
    costs = actualized_annual_costs(evaler=evaler,
                                    agentsId=agentsId)
    power_income = power_generated(evaler=evaler,
                                   agentsId=agentsId)
    power_income["Energy"] *= lcoe(evaler=evaler, agentsId=agentsId)

    base_col = ['AgentId', 'Year']
    added_col = base_col + ['Fuel',
                            'OperationMaintenance',
                            'Capital',
                            'Decommission']
    power_income = merge(power_income, base_col, costs, added_col)
    # I don't get this, basically we do lcoe*power - cost where lcoe = cost/power...
    # so benefit are necessary 0
    power_income['Capital'] = power_income['Energy'] - (power_income['Fuel'] +
                                                        power_income['OperationMaintenance'] +
                                                        power_income['Capital'] +
                                                        power_income['Decommission'])

    return power_income[['SimId', 'AgentId', 'Year', 'Capital']]


def child_benefit(evaler, agentsId=[]):
    """Input : sqlite output database and institution agent id
    Output : cumulative sum of income and expense (= - expenditures + income)
    """
    dfEntry = evaler.eval('AgentEntry').copy()
    childsId = eco_tools.get_child_id(agentsId, dfEntry)
    return benefit(evaler, childsId)


def all_benefit(evaler, agentsId=[]):
    """Input : sqlite output database and institution agent id
    Output : cumulative sum of income and expense (= - expenditures + income)
    """
    dfEntry = evaler.eval('AgentEntry').copy()
    agentsId += eco_tools.get_child_id(agentsId, dfEntry)
    return benefit(evaler, agentsId)


def lcoe(evaler, agentsId=[]):
    """Input : evaler database and agents id
    Output : Value corresponding to Levelized Cost of Electricity ($/MWh)
    """
    costs = actualized_annual_cost(evaler=evaler,
                                   agentsId=agentsId).drop(['AgentId', 'Year'], axis=1)
    power = power_generated(evaler=evaler, agentsId=agentsId)

    initialYear = evaler.eval('Info').loc[0, 'InitialYear']
    dfEntry = evaler.eval('AgentEntry')
    actualized_power = actualize_costs(df=power,
                                       columns=["Energy"],
                                       dfEntry=dfEntry,
                                       time_col="Year",
                                       time_factor=1.,
                                       t_0=initialYear).drop(['SimId', 'AgentId', 'Year'], axis=1)

    return (costs.sum(axis=1)).sum() / actualized_power.sum().sum()


def child_lcoe(evaler, agentsId=[]):
    """Input : evaler database and Instutions/Region id
    Output : Value corresponding to Levelized Cost of Electricity ($/MWh) of the child agents
    """
    dfEntry = evaler.eval('AgentEntry').copy()
    childsId = eco_tools.get_child_id(agentsId, dfEntry)
    return lcoe(evaler, agentsId=childsId)


def all_lcoe(evaler, agentsId=[]):
    """Input : evaler database and Instutions/Region id
    Output : Value corresponding to Levelized Cost of Electricity ($/MWh)
    """
    dfEntry = evaler.eval('AgentEntry').copy()
    agentsId += eco_tools.get_child_id(agentsId, dfEntry)
    return lcoe(evaler, agentsId=agentsId)


def period_costs(evaler, agentsId=[], t0=0, period=0, capital=True):
    """Input : evaler database, agents id, time window (t0, period)
    Output : cost at each time step t corresponding to actualized sum of
    expense in [t+t0, t+t0+period] divided by actualized power generated
    in [t+t0, t+t0+period]
    """

    costs = actualized_annual_cost(evaler, agentsId=agentsId, capital=capital)
    power = power_generated(evaler, agentsId)
    if t0 > 0 or period > 0:
        initialYear = evaler.eval('Info').loc[0, 'InitialYear']
        if (t0 > 0):
            costs = costs[costs['Year'] >= t0+initialYear]
            power = power[power['Year'] >= t0+initialYear]
        if period > 0:
            costs = costs[costs['Year'] <= t0+initialYear+period]
            power = power[power['Year'] <= t0+initialYear+period]
    base_col = ['AgentId', 'Year']
    added_col = base_col + ['Cost']
    costs = merge(power, base_col, costs, added_col)
    costs['Cost'] *= 1 / costs['Energy']

    return costs[['Year', 'AgentId', 'Cost']]


def child_period_costs(evaler, agentsId=[], t0=0, period=0, capital=True):
    """Input : evaler database, agents id, time window (t0, period)
    Output : cost at each time step t corresponding to actualized sum of
    expense in [t+t0, t+t0+period] divided by actualized power generated
    in [t+t0, t+t0+period]
    """
    dfEntry = evaler.eval('AgentEntry').copy()
    childsId = eco_tools.get_child_id(agentsId, dfEntry)
    return period_costs(evaler, agentsId=childsId, t0=t0, period=period, capital=capital)


def all_period_costs(evaler, agentsId=[], t0=0, period=0, capital=True):
    """Input : evaler database, agents id, time window (t0, period)
    Output : cost at each time step t corresponding to actualized sum of
    expense in [t+t0, t+t0+period] divided by actualized power generated
    in [t+t0, t+t0+period]
    """
    dfEntry = evaler.eval('AgentEntry').copy()
    agentsId += eco_tools.get_child_id(agentsId, dfEntry)
    return period_costs(evaler, agentsId=agentsId, t0=t0, period=period, capital=capital)


# Institution level


def average_lcoe(evaler, agentsId=[]):
    """Input : evaler database and a list of agents id
    Output : Variable cost corresponding at each time step (i.e. every year)
    to the weighted average of the reactors Levelized Cost of Electricity
    ($/MWh). A reactor is taken into account at a time step t only if it is
    active (i.e. already commissioned and not yet decommissioned) at this
    time step.
    """
    costs = actualized_annual_cost(evaler=evaler,
                                   agentsId=agentsId)
    power = power_generated(evaler=evaler, agentsId=agentsId)

    initialYear = evaler.eval('Info').loc[0, 'InitialYear']
    dfEntry = evaler.eval('AgentEntry')
    actualized_power = actualize_costs(df=power,
                                       columns=["Energy"],
                                       dfEntry=dfEntry,
                                       time_col="Year",
                                       time_factor=1.,
                                       t_0=initialYear)
    base_col = ['AgentId', 'Year']
    added_col = base_col + ['Cost']
    costs = merge(actualized_power, base_col, costs, added_col)
    costs["Cost"] *= 1 / costs["Energy"]
    costs.drop(["AgentId"], axis=1, inplace=True)
    costs = costs.groupby(['SimId', 'Year']).sum().reset_index()

    return costs[['SimId', 'Year', 'Cost']]


# Region level


def child_average_lcoe(evaler, agentsId=[]):
    """Input : evaler database and list of regions/Institution/Agent agent id
    Output : Variable cost corresponding at each time step (i.e. every year)
    to the weighted average of the reactors Levelized Cost of Electricity
    ($/MWh). A reactor is taken into account at a time step t if and only if
    it is active (i.e. already commissioned and not yet decommissioned) at
    this time step.
    """
    dfEntry = evaler.eval('AgentEntry')
    childId = eco_tools.get_child_id(agentsId, dfEntry)
    return average_lcoe(evaler, childId)


def all_average_lcoe(evaler, agentsId=[]):
    """Input : evaler database and list of regions/Institution/Agent agent id
    Output : Variable cost corresponding at each time step (i.e. every year)
    to the weighted average of the reactors Levelized Cost of Electricity
    ($/MWh). A reactor is taken into account at a time step t if and only if
    it is active (i.e. already commissioned and not yet decommissioned) at
    this time step.
    """
    dfEntry = evaler.eval('AgentEntry')
    agentsId += eco_tools.get_child_id(agentsId, dfEntry)
    return average_lcoe(evaler, agentsId)


# Simulation level


def simulation_actualized_annual_costs(evaler, capital=True):
    """Same as annual_cost except all values are actualized to the begin date
    of the SIMULATION
    """
    all_ids = evaler.eval('AgentEntry')['AgentId'].to_list()
    return all_actualized_annual_costs(evaler, agentsId=all_ids, capital=capital)


def simulation_annual_costs(evaler, capital=True, truncate=True):
    """Input : sqlite output database. It is possible not to take into account
    the construction costs (capital=False) if the reactors are supposed to
    have been built before the beginning of the simulation. It is also
    possible to truncate the simulation results and only have access to cash
    flows occurring between the two dates (begin and end) specified in
    'parameters.xml'. The truncation allows to let reactors decommission after
    the end of the simulation and thus to take into account cash flows that
    occur after the end of the simulation for example to calculate the LCOE.
    Output : total reactor costs per year over its lifetime at the simulation
    level.
    """
    all_ids = evaler.eval('AgentEntry')['AgentId'].to_list()
    return all_annual_costs(evaler, capital=capital)


def simulation_benefit(evaler):
    """Input : evaler database
    Output : cumulative sum of total income and total expense
            (= - expenditures + income) when all reactors of the simulation
            are taken into account
    """
    all_ids = evaler.eval('AgentEntry')['AgentId'].to_list()
    return all_benefit(evaler, agentsId=all_ids)


def simulation_period_costs(evaler, t0=0, period=20, capital=True):
    """Input : evaler database, time window (t0, period)
    Output : cost at each time step t corresponding to actualized sum of total
    expense in [t+t0, t+t0+period] divided by total actualized power generated
    in [t+t0, t+t0+period] when all reactors of the simulation are taken into
    account
    """
    all_ids = evaler.eval('AgentEntry')['AgentId'].to_list()
    return all_period_costs(evaler, agentsId=all_ids, t0=t0, period=period, capital=capital)


def simulation_power_generated(evaler):
    """Input : evaler database
    Output : Electricity generated in MWh every years by all the reactors of
    the simulation
    """
    all_ids = evaler.eval('AgentEntry')['AgentId'].to_list()
    return all_power_generated(evaler, agentsId=all_ids)


def simulation_lcoe(evaler):
    """Input : sqlite output database
    Output : Value corresponding to Levelized Cost of Electricity ($/MWh) when
    taking into account all reactors commissioned in the simulation
    """
    all_ids = evaler.eval('AgentEntry')['AgentId'].to_list()
    return all_lcoe(evaler)


def simulation_average_lcoe(evaler):
    """Input : evaler database and region agent id
    Output : Variable cost corresponding at each time step (i.e. every year)
    to the weighted average of the reactors Levelized Cost of Electricity
    ($/MWh). A reactor is taken into account at a time step t if and only if
    it is in activity (i.e. already commissioned and not yet decommissioned)
    at this time step.
    """
    all_ids = evaler.eval('AgentEntry')['AgentId'].to_list()
    return all_average_lcoe(evaler, agentsId=all_ids)
