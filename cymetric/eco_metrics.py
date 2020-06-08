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
    from cymetric.eco_tools import actualization_vector, isreactor, \
        capital_shape, get_filiation_per_id, eco_input_data
    from cymetric.evaluator import register_metric
except ImportError:
    # some wacky CI paths prevent absolute importing, try relative
    from .metrics import metric
    from .evaluator import register_metric
    from .eco_tools import capital_shape, isreactor, actualization_vector, \
        isreactor, get_filiation_per_id, eco_input_data
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
        filiation = get_filiation_per_id(agent_id, dfEntry)
        agent_eco_prop = eco_data.get_prototype_eco(filiation)

        deviation = float(agent_eco_prop["capital"]["deviation"])
        deviation *= np.random.randn(1)
        deviation = int(np.around(deviation))

        beforePeak = int(agent_eco_prop["capital"]["beforePeak"]) + deviation
        afterPeak = int(agent_eco_prop["capital"]["afterPeak"])
        cashFlowShape = capital_shape(beforePeak, afterPeak)

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
            cashFlowShape = capital_shape(
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


_eideps = ['AgentEntry']

_eischema = [('Prototype', ts.STRING),
             ('AgentId', ts.INT),
             ('ParentId', ts.INT),
             ('ReturnOnDebt', ts.DOUBLE),
             ('ReturnOnEquity', ts.DOUBLE),
             ('TaxRate', ts.DOUBLE),
             ('DiscountRate', ts.DOUBLE),
             ('beforePeak', ts.INT),
             ('afterPeak', ts.INT),
             ('constructionDuration', ts.INT),
             ('Capital_Deviation', ts.DOUBLE),
             ('OvernightCost', ts.DOUBLE),
             ('Duration', ts.INT),
             ('OvernightCost', ts.DOUBLE),
             ('FixedCost', ts.DOUBLE),
             ('VariableCost', ts.DOUBLE),
             ('OperationMaintenance_Deviation', ts.DOUBLE),
             ('FuelCommodity', ts.STRING),
             ('SupplyCost', ts.DOUBLE),
             ('WasteFee', ts.DOUBLE),
             ('Fuel_Deviation', ts.DOUBLE),
             ('Truncation_Begin', ts.INT),
             ('Truncation_End', ts.INT)]


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


def direct_annual_costs(evaler, agents=(), agentsId=(), capital=True):
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

    if len(agents) != 0:
        agentsId += dfEntry[dfEntry['Prototype'].isin(
            agents)]["AgentId"].tolist()

    childId = []
    while (childId != list(set(dfEntry[dfEntry['ParentId'].isin(
            agentsId)]['AgentId'].to_list()))):
        if len(agentsId) != 0:
            childId += dfEntry[dfEntry['ParentId'].isin(
                agentsId)]['AgentId'].to_list()
        childId = (list(set(childId)))
        agentsId += childId

    costs = direct_annual_costs(evaler, agentsId=childId)

    if not capital:
        del costs['Capital']
    costs = costs.groupby('Year').sum().reset_index()
    return costs


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

    if len(agents) != 0:
        agentsId += dfEntry[dfEntry['Prototype'].isin(
            agents)]["AgentId"].tolist()

    facilitiesId = agentsId
    while (facilitiesId != list(set(dfEntry[dfEntry['ParentId'].isin(
            agentsId)]['AgentId'].to_list() + agentsId))):
        if len(agentsId) != 0:
            facilitiesId += dfEntry[dfEntry['ParentId'].isin(
                agentsId)]['AgentId'].to_list()
        facilitiesId = list(set(facilitiesId))
        agentsId += facilitiesId

    costs = direct_annual_costs(evaler, agentsId=facilitiesId)

    if not capital:
        del costs['Capital']
    costs = costs.groupby('Year').sum().reset_index()
    return costs


def annual_costs_present_value(outputDb, reactorId=(), capital=True):
    """Same as annual_cost except all values are actualized to the begin date
    of the SIMULATION
    """
    costs = annual_costs(outputDb, reactorId, capital)
    actualization = actualization_vector(len(costs))
    actualization.index = costs.index
    return costs.apply(lambda x: x * actualization)


def average_cost(outputDb, reactorId, capital=True):
    """Input : sqlite output database, reactor's AgentId
    Output : value (in $/MWh) corresponding to the total costs (sum of annual
    costs) divided by the total power generated.
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfPower = evaler.eval('TimeSeriesPower').reset_index()
    powerGenerated = sum(
        dfPower[dfPower.AgentId == reactorId].loc[:, 'Value']) * 8760 / 12
    return annual_costs(outputDb, reactorId,
                        capital).sum().sum() / powerGenerated


def benefit(outputDb, reactorId):
    """Input : sqlite output database and reactor agent id
    Output : cumulative sum of actualized income and expense
             (= - expenditures + income)
    """
    costs = - annual_costs(outputDb, reactorId).sum(axis=1)
    powerGenerated = power_generated(
        outputDb, reactorId) * lcoe(outputDb, reactorId)
    rtn = pd.concat([costs, powerGenerated], axis=1).fillna(0)
    rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
    actualization = actualization_vector(len(rtn))
    actualization.index = rtn.index
    rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
    return rtn


def lcoe(outputDb, reactorId, capital=True):
    """Input : sqlite output database and reactor agent id
    Output : Value corresponding to Levelized Cost of Electricity ($/MWh)
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfEcoInfo = evaler.eval('EconomicInfo')
    dfEcoInfo = dfEcoInfo.set_index(('Agent', 'AgentId'))
    discountRate = dfEcoInfo.loc[reactorId, ('Finance', 'DiscountRate')]
    annualCosts = annual_costs(outputDb, reactorId, capital)
    powerGenerated = power_generated(outputDb, reactorId)
    actualization = actualization_vector(powerGenerated.size, discountRate)
    actualization.index = powerGenerated.index.copy()
    return (annualCosts.sum(axis=1) * actualization).fillna(0).sum() / \
        ((powerGenerated * actualization).fillna(0).sum())


def period_costs(outputDb, reactorId, t0=0, period=20, capital=True):
    """Input : sqlite output database, reactor id, time window (t0, period)
    Output : cost at each time step t corresponding to actualized sum of
    expense in [t+t0, t+t0+period] divided by actualized power generated
    in [t+t0, t+t0+period]
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo.loc[:, ('Truncation', 'Begin')]
    simulationEnd = dfEcoInfo.loc[:, ('Truncation', 'End')]
    costs = annual_costs(outputDb, reactorId, capital)
    costs = costs.sum(axis=1)
    power = power_generated(outputDb, reactorId)
    df = pd.DataFrame(index=list(
        range(initialYear, initialYear + duration // 12 + 1)))
    df['Power'] = power
    df['Costs'] = costs
    df = df.fillna(0)
    simulationBegin = (simulationBegin + initialMonth -
                       1) // 12 + initialYear  # year instead of months
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd)))
    rtn['Power'] = pd.Series()
    rtn['Payment'] = pd.Series()
    rtn = rtn.fillna(0)
    for i in range(simulationBegin + t0, simulationBegin + t0 + period):
        rtn.loc[simulationBegin, 'Power'] += df.loc[i, 'Power'] / \
            (1 + default_discount_rate) ** (i - simulationBegin)
        rtn.loc[simulationBegin, 'Payment'] += df.loc[i, 'Costs'] / \
            (1 + default_discount_rate) ** (i - simulationBegin)
    for j in range(simulationBegin + 1, simulationEnd):
        rtn.loc[j, 'Power'] = rtn.loc[j - 1, 'Power'] * \
            (1 + default_discount_rate) - \
            df.loc[j - 1 + t0, 'Power'] * \
            (1 + default_discount_rate) ** (1 - t0) + \
            df.loc[j - 1 + period + t0, 'Power'] / \
            (1 + default_discount_rate) ** (period + t0 - 1)
        rtn.loc[j, 'Payment'] = rtn.loc[j - 1, 'Payment'] * \
            (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Costs'] *\
            (1 + default_discount_rate) ** (1 - t0) + \
            df.loc[j - 1 + period + t0, 'Costs'] / \
            (1 + default_discount_rate) ** (period + t0 - 1)
    rtn['Ratio'] = rtn['Payment'] / rtn['Power'] * (rtn['Power'] > 1)
    actualization = actualization_vector(len(rtn))
    actualization.index = rtn.index
    rtn['Actualized'] = rtn['Ratio'] * actualization
    return rtn


def period_costs2(outputDb, reactorId, t0=0, period=20, capital=True):
    """Just for tests : another way to calculate the period costs
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    costs = annual_costs(outputDb, institutionId, capital)
    costs = costs.sum(axis=1)
    power = power_generated(outputDb, institutionId)
    df = pd.DataFrame(
        index=list(
            range(
                initialYear,
                initialYear +
                duration //
                12 +
                1)))
    df['Power'] = power
    df['Costs'] = costs
    df = df.fillna(0)
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo.loc[:, ('Truncation', 'Begin')]
    simulationEnd = dfEcoInfo.loc[:, ('Truncation', 'End')]
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
    rtn['Power'] = pd.Series()
    rtn['Payment'] = pd.Series()
    rtn = rtn.fillna(0)
    for j in range(simulationBegin, simulationEnd + 1):
        for i in range(j + t0, j + t0 + period):
            rtn.loc[j, 'Power'] += df.loc[i, 'Power'] / \
                (1 + default_discount_rate) ** (i - j)
            rtn.loc[j, 'Payment'] += df.loc[i, 'Costs'] / \
                (1 + default_discount_rate) ** (i - j)
            # tmp['WasteManagement'][j] = pd.Series()
    rtn['Ratio'] = rtn['Payment'] / rtn['Power'] * (rtn['Power'] > 1)
    return rtn


def power_generated(outputDb, reactorId):
    """Input : sqlite output database and reactor agent id
    Output : Electricity generated in MWh every year
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfPower = evaler.eval('TimeSeriesPower').reset_index()
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfPower = dfPower[dfPower['AgentId'] == reactorId].copy()
    dfPower['Year'] = (dfPower['Time'] + initialMonth - 1) // 12 + initialYear
    dfPower = dfPower.groupby('Year').sum()
    rtn = pd.Series(dfPower['Value'] * 8760 / 12,
                    index=list(range(initialYear,
                                     initialYear + (initialMonth + duration) //
                                     12 + 1)))
    return rtn.fillna(0)

# Institution level


def institution_annual_costs_present_value(outputDb, reactorId, capital=True):
    """Same as annual_cost except all values are actualized to the begin date
    of the SIMULATION
    """
    costs = institution_annual_costs(outputDb, institutionId, capital)
    actualization = actualization_vector(len(costs))
    actualization.index = costs.index
    return costs.apply(lambda x: x * actualization)


def institution_benefit(outputDb, institutionId):
    """Input : sqlite output database and institution agent id
    Output : cumulative sum of income and expense (= - expenditures + income)
    """
    costs = - institution_annual_costs(outputDb, institutionId).sum(axis=1)
    power_gen = institution_power_generated(
        outputDb, institutionId) * institution_average_lcoe(
        outputDb, institutionId)['Average LCOE']
    rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
    rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
    actualization = actualization_vector(len(rtn))
    actualization.index = rtn.index
    rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
    return rtn


def institution_period_costs(
        outputDb, institutionId, t0=0, period=20, capital=True):
    """Input : sqlite output database, institution id, time window (t0, period)
    Output : cost at each time step t corresponding to actualized sum of
    expense in [t+t0, t+t0+period] divided by actualized power generated in
    [t+t0, t+t0+period]
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEconomicInfo = evaler.eval('EconomicInfo').reset_index()
    simulationBegin = dfEconomicInfo.loc[:, ('Truncation', 'Begin')]
    simulationEnd = dfEconomicInfo.loc[:, ('Truncation', 'End')]
    costs = institution_annual_costs(
        outputDb, institutionId, capital, truncate=False)
    costs = costs.sum(axis=1)
    power = institution_power_generated(
        outputDb, institutionId, truncate=False)
    df = pd.DataFrame(
        index=list(
            range(
                initialYear,
                initialYear + duration // 12 + 1)))
    df['Power'] = power
    df['Costs'] = costs
    df = df.fillna(0)
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd)))
    rtn['Power'] = pd.Series()
    rtn['Payment'] = pd.Series()
    rtn = rtn.fillna(0)
    for i in range(simulationBegin + t0, simulationBegin + t0 + period):
        rtn.loc[simulationBegin, 'Power'] += df.loc[i, 'Power'] / \
            (1 + default_discount_rate) ** (i - simulationBegin)
        rtn.loc[simulationBegin, 'Payment'] += df.loc[i, 'Costs'] / \
            (1 + default_discount_rate) ** (i - simulationBegin)
    for j in range(simulationBegin + 1, simulationEnd):
        rtn.loc[j, 'Power'] = rtn.loc[j - 1, 'Power'] * \
            (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Power'] * \
            (1 + default_discount_rate) ** (1 - t0) + \
            df.loc[j - 1 + period + t0, 'Power'] / \
            (1 + default_discount_rate) ** (period + t0 - 1)
        rtn.loc[j, 'Payment'] = rtn.loc[j - 1, 'Payment'] * \
            (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Costs'] * \
            (1 + default_discount_rate) ** (1 - t0) + \
            df.loc[j - 1 + period + t0, 'Costs'] / \
            (1 + default_discount_rate) ** (period + t0 - 1)
    rtn['Ratio'] = rtn['Payment'] / rtn['Power'] * (rtn['Power'] > 1)
    actualization = actualization_vector(len(rtn))
    actualization.index = rtn.index
    rtn['Actualized'] = rtn['Ratio'] * actualization
    return rtn


def institution_period_costs2(
        outputDb, institutionId, t0=0, period=20, capital=True):
    """Just for tests : another method for period costs
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEconomicInfo = evaler.eval('EconomicInfo').reset_index()
    simulationBegin = dfEconomicInfo.loc[:, ('Truncation', 'Begin')]
    simulationEnd = dfEconomicInfo.loc[:, ('Truncation', 'End')]
    costs = institution_annual_costs(
        outputDb, institutionId, capital, truncate=False)
    costs = costs.sum(axis=1)
    power = institution_power_generated(
        outputDb, institutionId, truncate=False)
    df = pd.DataFrame(
        index=list(
            range(
                initialYear,
                initialYear +
                duration //
                12 +
                1)))
    df['Power'] = power
    df['Costs'] = costs
    df = df.fillna(0)
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
    rtn['Power'] = pd.Series()
    rtn['Payment'] = pd.Series()
    rtn = rtn.fillna(0)
    for j in range(simulationBegin, simulationEnd + 1):
        for i in range(j + t0, j + t0 + period):
            rtn.loc[j, 'Power'] += df.loc[i, 'Power'] / \
                (1 + default_discount_rate) ** (i - j)
            rtn.loc[j, 'Payment'] += df.loc[i, 'Costs'] / \
                (1 + default_discount_rate) ** (i - j)
    rtn['Ratio'] = rtn['Payment'] / rtn['Power'] * (rtn['Power'] > 1)
    return rtn


def institution_power_generated(outputDb, institutionId, truncate=True):
    """Input : sqlite output database and institution agent id
    Output : Sum of electricity generated in MWh every year in the institution
    reactor fleet
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
    simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
    dfEntry = evaler.eval('AgentEntry').reset_index()
    dfEntry = dfEntry[dfEntry.ParentId == institutionId]
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(
        lambda x: x > simulationBegin and x < simulationEnd)]
    dfPower = evaler.eval('TimeSeriesPower')
    reactorIds = dfEntry[dfEntry['AgentId'].apply(
        lambda x: isreactor(dfPower, x))]['AgentId'].tolist()
    dfPower = evaler.eval('TimeSeriesPower').reset_index()
    dfPower = dfPower[dfPower['AgentId'].apply(lambda x: x in reactorIds)]
    dfPower['Year'] = (dfPower['Time'] + initialMonth - 1) // 12 + initialYear
    dfPower = dfPower.groupby('Year').sum()
    rtn = pd.Series(dfPower['Value'] * 8760 / 12,
                    index=list(range(initialYear,
                                     initialYear + (initialMonth + duration) //
                                     12 + 1)))
    rtn.name = 'Power in MWh'
    return rtn.fillna(0)


def institution_lcoe(outputDb, institutionId):
    """Input : sqlite output database and institution agent id
    Output : Value corresponding to Levelized Cost of Electricity ($/MWh)
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfEcoInfo = evaler.eval('EconomicInfo')
    dfEcoInfo = dfEcoInfo.set_index(('Agent', 'AgentId'))
    discountRate = dfEcoInfo.loc[institutionId, ('Finance', 'DiscountRate')]
    annualCosts = institution_annual_costs(outputDb, institutionId)
    powerGenerated = institution_power_generated(outputDb, institutionId)
    actualization = actualization_vector(powerGenerated.size, discountRate)
    actualization.index = powerGenerated.index.copy()
    return (annualCosts.sum(axis=1) * actualization).fillna(0).sum() / \
        ((powerGenerated * actualization).fillna(0).sum())


def institution_average_lcoe(outputDb, institutionId):
    """Input : sqlite output database and institution agent id
    Output : Variable cost corresponding at each time step (i.e. every year)
    to the weighted average of the reactors Levelized Cost of Electricity
    ($/MWh). A reactor is taken into account at a time step t only if it is
    active (i.e. already commissioned and not yet decommissioned) at this
    time step.
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
    simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
    dfEntry = evaler.eval('AgentEntry').reset_index()
    dfEntry = dfEntry[dfEntry.ParentId == institutionId]
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(
        lambda x: x > simulationBegin and x < simulationEnd)]
    dfPower = evaler.eval('TimeSeriesPower')
    reactorIds = dfEntry[dfEntry['AgentId'].apply(
        lambda x: isreactor(dfPower, x))]['AgentId'].tolist()
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    dfPower = evaler.eval('TimeSeriesPower')
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
    rtn['Weighted sum'] = 0
    rtn['Power'] = 0
    rtn['Temp'] = pd.Series()
    rtn['Temp2'] = pd.Series()
    for id in reactorIds:
        tmp = lcoe(outputDb, id)
        commissioning = dfEntry[dfEntry.AgentId == id]['EnterTime'].iloc[0]
        lifetime = dfEntry[dfEntry.AgentId == id]['Lifetime'].iloc[0]
        decommissioning = (commissioning + lifetime +
                           initialMonth - 1) // 12 + initialYear
        commissioning = (commissioning + initialMonth - 1) // 12 + initialYear
        power = dfPower[dfPower.AgentId == id]['Value'].iloc[0]
        rtn['Temp'] = pd.Series(
            tmp,
            index=list(
                range(
                    commissioning,
                    decommissioning + 1))) * power
        rtn['Weighted sum'] += rtn['Temp'].fillna(0)
        rtn['Temp2'] = pd.Series(
            power,
            index=list(
                range(commissioning,
                      decommissioning + 1))).fillna(0)
        rtn['Power'] += rtn['Temp2'].fillna(0)
    rtn['Average LCOE'] = rtn['Weighted sum'] / rtn['Power']
    return rtn.fillna(0)

# Region level


def region_annual_costs_present_value(
        outputDb, regionId, capital=True, truncate=True):
    """Same as annual_cost except all values are actualized to the begin date
    of the SIMULATION
    """
    costs = region_annual_costs(outputDb, regionId, capital)
    actualization = actualization_vector(len(costs))
    actualization.index = costs.index
    return costs.apply(lambda x: x * actualization)


def region_benefit(outputDb, regionId):
    """Input : sqlite output database and region agent id
    Output : cumulative sum of actualized income and expense
            (= - expenditures + income)
    """
    costs = - region_annual_costs(outputDb, regionId).sum(axis=1)
    power_gen = region_power_generated(outputDb, regionId) * \
        region_average_lcoe(outputDb, regionId)['Average LCOE']
    rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
    rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
    actualization = actualization_vector(len(rtn))
    actualization.index = rtn.index
    rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
    return rtn


def region_period_costs(outputDb, regionId, t0=0, period=20, capital=True):
    """Input : sqlite output database, region id, time window (t0, period)
    Output : cost at each time step t corresponding to actualized sum of
    expense in [t+t0, t+t0+period] divided by actualized power generated
    in [t+t0, t+t0+period]
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
    simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
    costs = region_annual_costs(outputDb, regionId, capital, truncate=False)
    costs = costs.sum(axis=1)
    power = region_power_generated(outputDb, regionId, truncate=False)
    df = pd.DataFrame(
        index=list(
            range(
                initialYear,
                initialYear +
                duration //
                12 +
                1)))
    df['Power'] = power
    df['Costs'] = costs
    df = df.fillna(0)
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
    rtn['Power'] = pd.Series()
    rtn['Payment'] = pd.Series()
    rtn = rtn.fillna(0)
    for i in range(simulationBegin + t0, simulationBegin + t0 + period):
        rtn.loc[simulationBegin, 'Power'] += df.loc[i, 'Power'] / \
            (1 + default_discount_rate) ** (i - simulationBegin)
        rtn.loc[simulationBegin, 'Payment'] += df.loc[i, 'Costs'] / \
            (1 + default_discount_rate) ** (i - simulationBegin)
    for j in range(simulationBegin + 1, simulationEnd + 1):
        rtn.loc[j, 'Power'] = rtn.loc[j - 1, 'Power'] * \
            (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Power'] * \
            (1 + default_discount_rate) ** (1 - t0) + \
            df.loc[j - 1 + period + t0, 'Power'] / \
            (1 + default_discount_rate) ** (period + t0 - 1)
        rtn.loc[j, 'Payment'] = rtn.loc[j - 1, 'Payment'] * \
            (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Costs'] * \
            (1 + default_discount_rate) ** (1 - t0) + \
            df.loc[j - 1 + period + t0, 'Costs'] / \
            (1 + default_discount_rate) ** (period + t0 - 1)
    rtn['Ratio'] = rtn['Payment'] / rtn['Power'] * (rtn['Power'] > 1)
    actualization = actualization_vector(len(rtn))
    actualization.index = rtn.index
    rtn['Actualized'] = rtn['Ratio'] * actualization
    return rtn


def region_period_costs2(outputDb, regionId, t0=0, period=20, capital=True):
    """Just for tests : another method to calculate period costs
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
    simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
    costs = region_annual_costs(outputDb, regionId, capital, truncate=False)
    costs = costs.sum(axis=1)
    power = region_power_generated(outputDb, regionId, truncate=False)
    df = pd.DataFrame(
        index=list(
            range(
                initialYear,
                initialYear +
                duration //
                12 +
                1)))
    df['Power'] = power
    df['Costs'] = costs
    df = df.fillna(0)
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
    rtn['Power'] = pd.Series()
    rtn['Payment'] = pd.Series()
    rtn = rtn.fillna(0)
    for j in range(simulationBegin, simulationEnd + 1):
        for i in range(j + t0, j + t0 + period):
            rtn.loc[j, 'Power'] += df.loc[i, 'Power'] / \
                (1 + default_discount_rate) ** (i - j)
            rtn.loc[j, 'Payment'] += df.loc[i, 'Costs'] / \
                (1 + default_discount_rate) ** (i - j)
    rtn['Ratio'] = rtn['Payment'] / rtn['Power'] * (rtn['Power'] > 1)
    return rtn


def region_power_generated(outputDb, regionId, truncate=True):
    """Input : sqlite output database and region agent id
    Output : Sum of electricity generated in MWh every years in the region
    reactor fleet
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
    simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
    dfEntry = evaler.eval('AgentEntry').reset_index()
    tmp = dfEntry[dfEntry.ParentId == regionId]
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(
        lambda x: x > simulationBegin and x < simulationEnd)]
    institutionIds = tmp[tmp.Kind == 'Inst']['AgentId'].tolist()
    reactorIds = []
    for id in institutionIds:
        dfEntry2 = dfEntry[dfEntry.ParentId == id]
        reactorIds += dfEntry2[dfEntry2['Spec'].apply(
            lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    dfPower = evaler.eval('TimeSeriesPower').reset_index()
    dfPower = dfPower[dfPower['AgentId'].apply(lambda x: x in reactorIds)]
    dfPower['Year'] = (dfPower['Time'] + initialMonth - 1) // 12 + initialYear
    dfPower = dfPower.groupby('Year').sum()
    rtn = pd.Series(dfPower['Value'] * 8760 / 12,
                    index=list(range(initialYear,
                                     initialYear + (initialMonth + duration) //
                                     12 + 1)))
    rtn.name = 'Power in MWh'
    return rtn.fillna(0)


def region_lcoe(outputDb, regionId):
    """Input : sqlite output database and region agent id
    Output : Value corresponding to Levelized Cost of Electricity ($/MWh)
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfEcoInfo = evaler.eval('EconomicInfo')
    dfEcoInfo = dfEcoInfo.set_index(('Agent', 'AgentId'))
    discountRate = dfEcoInfo.loc[regionId, ('Finance', 'DiscountRate')]
    annualCosts = region_annual_costs(outputDb, regionId)
    powerGenerated = region_power_generated(outputDb, regionId)
    actualization = actualization_vector(powerGenerated.size, discountRate)
    actualization.index = powerGenerated.index.copy()
    return (annualCosts.sum(axis=1) * actualization).fillna(0).sum() / \
        ((powerGenerated * actualization).fillna(0).sum())


def region_average_lcoe(outputDb, regionId):
    """Input : sqlite output database and region agent id
    Output : Variable cost corresponding at each time step (i.e. every year)
    to the weighted average of the reactors Levelized Cost of Electricity
    ($/MWh). A reactor is taken into account at a time step t if and only if
    it is active (i.e. already commissioned and not yet decommissioned) at
    this time step.
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
    simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
    dfEntry = evaler.eval('AgentEntry').reset_index()
    tmp = dfEntry[dfEntry.ParentId == regionId]
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(
        lambda x: x > simulationBegin and x < simulationEnd)]
    institutionsId = tmp[tmp.Kind == 'Inst']['AgentId'].tolist()
    reactorIds = []
    dfPower = evaler.eval('TimeSeriesPower')
    for id in institutionsId:
        dfEntry2 = dfEntry[dfEntry.ParentId == id]
        reactorIds += dfEntry2[dfEntry2['Spec'].apply(
            lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
    rtn['Weighted sum'] = 0
    rtn['Power'] = 0
    rtn['Temp'] = pd.Series()
    rtn['Temp2'] = pd.Series()
    for id in reactorIds:
        tmp = lcoe(outputDb, id)
        commissioning = dfEntry[dfEntry.AgentId == id]['EnterTime'].iloc[0]
        lifetime = dfEntry[dfEntry.AgentId == id]['Lifetime'].iloc[0]
        decommissioning = (commissioning + lifetime +
                           initialMonth - 1) // 12 + initialYear
        commissioning = (commissioning + initialMonth - 1) // 12 + initialYear
        power = dfPower[dfPower.AgentId == id]['Value'].iloc[0]
        rtn['Temp'] = pd.Series(
            tmp,
            index=list(
                range(
                    commissioning,
                    decommissioning + 1))) * power
        rtn['Weighted sum'] += rtn['Temp'].fillna(0)
        rtn['Temp2'] = pd.Series(
            power,
            index=list(
                range(
                    commissioning,
                    decommissioning +
                    1))).fillna(0)
        rtn['Power'] += rtn['Temp2']
    rtn['Average LCOE'] = rtn['Weighted sum'] / rtn['Power']
    return rtn.fillna(0)

# Simulation level


def simulation_annual_costs(outputDb, capital=True, truncate=True):
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
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
    simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
    dfEntry = evaler.eval('AgentEntry').reset_index()
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(
        lambda x: x > simulationBegin and x < simulationEnd)]
    reactorIds = dfEntry[dfEntry['Spec'].apply(
        lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    dfCapitalCosts = evaler.eval('CapitalCost').reset_index()
    dfCapitalCosts = dfCapitalCosts[dfCapitalCosts['AgentId'].apply(
        lambda x: x in reactorIds)]
    mini = min(dfCapitalCosts['Time'])
    dfCapitalCosts = dfCapitalCosts.groupby('Time').sum()
    costs = pd.DataFrame(
        {'Capital': dfCapitalCosts['Payment']}, index=list(range(0, duration)))
    dfDecommissioningCosts = evaler.eval('DecommissioningCost').reset_index()
    if not dfDecommissioningCosts.empty:
        dfDecommissioningCosts = dfDecommissioningCosts[
            dfDecommissioningCosts['AgentId'].apply(lambda x: x in reactorIds)]
        dfDecommissioningCosts = dfDecommissioningCosts.groupby('Time').sum()
        costs['Decommissioning'] = dfDecommissioningCosts['Payment']
    dfOMCosts = evaler.eval('OperationMaintenance').reset_index()
    dfOMCosts = dfOMCosts[dfOMCosts['AgentId'].apply(
        lambda x: x in reactorIds)]
    dfOMCosts = dfOMCosts.groupby('Time').sum()
    costs['OperationAndMaintenance'] = dfOMCosts['Payment']
    dfFuelCosts = evaler.eval('FuelCost').reset_index()
    dfFuelCosts = dfFuelCosts[dfFuelCosts['AgentId'].apply(
        lambda x: x in reactorIds)]
    dfFuelCosts = dfFuelCosts.groupby('Time').sum()
    costs['Fuel'] = dfFuelCosts['Payment']
    costs = costs.fillna(0)
    costs['Year'] = (costs.index + initialMonth - 1) // 12 + initialYear
    if truncate:
        endYear = (simulationEnd + initialMonth - 1) // 12 + initialYear
        costs = costs[costs['Year'].apply(lambda x: x <= endYear)]
        beginYear = (simulationBegin + initialMonth - 1) // 12 + initialYear
        costs = costs[costs['Year'].apply(lambda x: x >= beginYear)]
    if not capital:
        del costs['Capital']
    costs = costs.groupby('Year').sum()
    return costs


def simulation_annual_costs_present_value(
        outputDb, capital=True, truncate=True):
    """Same as annual_cost except all values are actualized to the begin date
    of the SIMULATION
    """
    df = simulation_annual_costs(outputDb, capital, truncate)
    for year in df.index:
        df.loc[year, :] = df.loc[year, :] / \
            (1 + default_discount_rate) ** (year - df.index[0])
    return df


def simulation_benefit(outputDb):
    """Input : sqlite output database
    Output : cumulative sum of total income and total expense
            (= - expenditures + income) when all reactors of the simulation
            are taken into account
    """
    costs = - simulation_annual_costs(outputDb).sum(axis=1)
    power_gen = simulation_power_generated(
        outputDb) * simulation_average_lcoe(outputDb)['Average LCOE']
    rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
    rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
    actualization = actualization_vector(len(rtn))
    actualization.index = rtn.index
    rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
    return rtn


def simulation_period_costs(outputDb, t0=0, period=20, capital=True):
    """Input : sqlite output database, time window (t0, period)
    Output : cost at each time step t corresponding to actualized sum of total
    expense in [t+t0, t+t0+period] divided by total actualized power generated
    in [t+t0, t+t0+period] when all reactors of the simulation are taken into
    account
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
    simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
    costs = simulation_annual_costs(
        outputDb,
        capital,
        truncate=False).sum(
        axis=1)
    power = simulation_power_generated(outputDb, truncate=False)
    df = pd.DataFrame(
        index=list(
            range(
                initialYear,
                initialYear +
                duration //
                12 +
                1)))
    df['Power'] = power
    df['Costs'] = costs
    df = df.fillna(0)
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd)))
    rtn['Power'] = pd.Series()
    rtn['Payment'] = pd.Series()
    rtn = rtn.fillna(0)
    for i in range(simulationBegin + t0, simulationBegin + t0 + period):
        rtn.loc[simulationBegin, 'Power'] += df.loc[i, 'Power'] / \
            (1 + default_discount_rate) ** (i - simulationBegin)
        rtn.loc[simulationBegin, 'Payment'] += df.loc[i, 'Costs'] / \
            (1 + default_discount_rate) ** (i - simulationBegin)
    for j in range(simulationBegin + 1, simulationEnd):
        rtn.loc[j, 'Power'] = rtn.loc[j - 1, 'Power'] * \
            (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Power'] * \
            (1 + default_discount_rate) ** (1 - t0) + \
            df.loc[j - 1 + period + t0, 'Power'] / \
            (1 + default_discount_rate) ** (period + t0 - 1)
        rtn.loc[j, 'Payment'] = rtn.loc[j - 1, 'Payment'] * \
            (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Costs'] * \
            (1 + default_discount_rate) ** (1 - t0) + \
            df.loc[j - 1 + period + t0, 'Costs'] / \
            (1 + default_discount_rate) ** (period + t0 - 1)
    rtn['Ratio'] = rtn['Payment'] / rtn['Power'] * (rtn['Power'] > 1)
    return rtn


def simulation_period_costs2(outputDb, t0=0, period=20, capital=True):
    """Just for tests : another method to calculate the period costs
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
    simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
    costs = simulation_annual_costs(
        outputDb,
        capital,
        truncate=False).sum(
        axis=1)
    power = simulation_power_generated(outputDb, truncate=False)
    df = pd.DataFrame(
        index=list(
            range(
                initialYear,
                initialYear +
                duration //
                12 +
                1)))
    df['Power'] = power
    df['Costs'] = costs
    df = df.fillna(0)
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
    rtn['Power'] = pd.Series()
    rtn['Payment'] = pd.Series()
    rtn = rtn.fillna(0)
    for j in range(simulationBegin, simulationEnd + 1):
        for i in range(j + t0, j + t0 + period):
            rtn.loc[j, 'Power'] += df.loc[i, 'Power'] / \
                (1 + default_discount_rate) ** (i - j)
            rtn.loc[j, 'Payment'] += df.loc[i, 'Costs'] / \
                (1 + default_discount_rate) ** (i - j)
    rtn['Ratio'] = rtn['Payment'] / rtn['Power'] * (rtn['Power'] > 1)
    return rtn


def simulation_power_generated(outputDb, truncate=True):
    """Input : sqlite output database
    Output : Electricity generated in MWh every years by all the reactors of
    the simulation
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
    simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
    dfEntry = evaler.eval('AgentEntry').reset_index()
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(
        lambda x: x > simulationBegin and x < simulationEnd)]
    reactorIds = dfEntry[dfEntry['Spec'].apply(
        lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    dfPower = evaler.eval('TimeSeriesPower').reset_index()
    dfPower = dfPower[dfPower['AgentId'].apply(lambda x: x in reactorIds)]
    dfPower['Year'] = (dfPower['Time'] + initialMonth - 1) // 12 + initialYear
    dfPower = dfPower.groupby('Year').sum()
    rtn = pd.Series(dfPower['Value'] * 8760 / 12,
                    index=list(range(initialYear,
                                     initialYear + (initialMonth + duration) //
                                     12 + 1)))
    rtn.name = 'Power in MWh'
    return rtn.fillna(0)


def simulation_lcoe(outputDb):
    """Input : sqlite output database
    Output : Value corresponding to Levelized Cost of Electricity ($/MWh) when
    taking into account all reactors commissioned in the simulation
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfEcoInfo = evaler.eval('EconomicInfo')
    dfEcoInfo = dfEcoInfo.set_index(('Agent', 'AgentId'))
    discountRate = dfEcoInfo.iloc[0, ('Finance', 'DiscountRate')]
    annualCosts = simulation_annual_costs(outputDb)
    powerGenerated = simulation_power_generated(outputDb)
    actualization = actualization_vector(powerGenerated.size, discountRate)
    actualization.index = powerGenerated.index.copy()
    return (annualCosts.sum(axis=1) * actualization).fillna(0).sum() / \
        ((powerGenerated * actualization).fillna(0).sum())


def simulation_average_lcoe(outputDb):
    """Input : sqlite output database and region agent id
    Output : Variable cost corresponding at each time step (i.e. every year)
    to the weighted average of the reactors Levelized Cost of Electricity
    ($/MWh). A reactor is taken into account at a time step t if and only if
    it is in activity (i.e. already commissioned and not yet decommissioned)
    at this time step.
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEcoInfo = evaler.eval('EconomicInfo')
    simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
    simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
    dfEntry = evaler.eval('AgentEntry').reset_index()
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(
        lambda x: x > simulationBegin and x < simulationEnd)]
    reactorIds = dfEntry[dfEntry['Spec'].apply(
        lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    dfPower = evaler.eval('TimeSeriesPower')
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
    rtn['Weighted sum'] = 0
    rtn['Power'] = 0
    rtn['Temp'] = pd.Series()
    rtn['Temp2'] = pd.Series()
    for id in reactorIds:
        tmp = lcoe(outputDb, id)
        commissioning = dfEntry[dfEntry.AgentId == id]['EnterTime'].iloc[0]
        lifetime = dfEntry[dfEntry.AgentId == id]['Lifetime'].iloc[0]
        decommissioning = (commissioning + lifetime +
                           initialMonth - 1) // 12 + initialYear
        commissioning = (commissioning + initialMonth - 1) // 12 + initialYear
        power = dfPower[dfPower.AgentId == id]['Value'].iloc[0]
        rtn['Temp'] = pd.Series(
            tmp,
            index=list(
                range(
                    commissioning,
                    decommissioning + 1))) * power
        rtn['Weighted sum'] += rtn['Temp'].fillna(0)
        rtn['Temp2'] = pd.Series(
            power,
            index=list(
                range(
                    commissioning,
                    decommissioning +
                    1)))
        rtn['Power'] += rtn['Temp2'].fillna(0)
    rtn['Average LCOE'] = rtn['Weighted sum'] / rtn['Power']
    return rtn.fillna(0)
