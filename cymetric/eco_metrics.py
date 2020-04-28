##### economic metrics for nuclear power plants #####

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
    from cymetric.eco_tools import actualization_vector, isreactor, capital_shape
    from cymetric.evaluator import register_metric
except ImportError:
    # some wacky CI paths prevent absolute importing, try relative
    from .metrics import metric
    from .evaluator import register_metric
    from .eco_tools import capital_shape, isreactor, actualization_vector, isreactor
    from .tools import dbopen, merge
    from .evaluator import Evaluator

xml_inputs = 'parameters.xml' # This xml file has to be built in the same direction as the sqlite output database. It contains the economic data needed to calculate the EconomicInfo metric

## The actual metrics ##


_ccdeps = ['TimeSeriesPower', 'AgentEntry', 'Info', 'EconomicInfo']

_ccschema = [
    ('SimId', ts.UUID),
    ('AgentId', ts.INT),
    ('Time', ts.INT),
    ('Payment', ts.DOUBLE)]


@metric(name='CapitalCost', depends=_ccdeps, schema=_ccschema)
def capital_cost(dfPower, dfEntry, dfInfo, dfEcoInfo):
    """The CapitalCost metric gives the cash flows at each time step corresponding to the reactors construction costs.
    """

    simDuration = dfInfo['Duration'].iloc[0]
    dfEntry = pd.DataFrame([dfEntry.EnterTime, dfEntry.AgentId]).transpose()
    #dfEntry = dfEntry.set_index(['AgentId'])
    rtn = pd.DataFrame()
    for index, eco_row in dfEcoInfo.iterrows():
        id = eco_row['AgentId']
        if isreactor(dfPower, id):
            deviation = eco_row['Capital_Deviation'] * np.random.randn(1)
            deviation = int(np.around(deviation))

            beforePeak = int(eco_row['Capital_beforePeak'] + deviation)
            afterPeak = int(eco_row['Capital_afterPeak'])

            entry_time = dfEntry[dfEntry['AgentId'] == id].EnterTime.values[0]
            constructionDuration = int(
                eco_row['Capital_constructionDuration'] + deviation)
            powerCapacity = max(dfPower[dfPower.AgentId == id]['Value'])

            overnightCost = eco_row['Capital_OvernightCost']
            cashFlowShape = capital_shape(beforePeak, afterPeak)
            discountRate = eco_row['Finance_DiscountRate']
            cashFlow = np.around(
                cashFlowShape * overnightCost * powerCapacity, 4)
            cashFlow *= ((1 + discountRate) ** math.ceil((beforePeak + afterPeak) / 12) -
                         1) / (discountRate * math.ceil((beforePeak + afterPeak) / 12))

            tmp = pd.DataFrame(data={'AgentId': id,
                                     'Time': pd.Series(list(range(beforePeak + afterPeak + 1)))
                                     + entry_time - constructionDuration,
                                     'Payment': cashFlow},
                               columns=['AgentId', 'Time', 'Payment'])
            rtn = pd.concat([rtn, tmp], ignore_index=False)

    rtn['SimId'] = dfPower['SimId'].iloc[0]
    rtn = rtn[rtn['Time'].apply(lambda x: x >= 0 and x < simDuration)]
    rtn = rtn.reset_index()
    return rtn[['SimId', 'AgentId', 'Time', 'Payment']]


del _ccdeps, _ccschema


_fcdeps = ['Resources', 'Transactions', 'EconomicInfo']

_fcschema = [
    ('SimId', ts.UUID),
    ('TransactionId', ts.INT),
    ('AgentId', ts.INT),
    ('Commodity', ts.STRING),
    ('Payment', ts.DOUBLE),
    ('Time', ts.INT)]

@metric(name='FuelCost', depends=_fcdeps, schema=_fcschema)
def fuel_cost(dfResources, dfTransactions, dfEcoInfo):
    """The FuelCost metric gives the cash flows at each time step corresponding to the reactors fuel costs. It also contains the waste fee.
    """

    # Unsure if it is about Sender or Receiver implementation here and test are not in agreement, taking receiver (using implementation as ref)
    rtn = dfTransactions.rename(columns={'ReceiverId': 'AgentId'})

    # add quantity to Transaction
    base_col = ['SimId', 'ResourceId']
    added_col = base_col + ['Quantity']
    rtn = merge(rtn, base_col, dfResources, added_col)

    # Merge Eco with Transaction per ReceiverId and commodity
    base_col = ['AgentId', 'Commodity']
    # , 'Finance_DiscountRate']
    added_col = base_col + ['Fuel_SupplyCost',
                            'Fuel_WasteFee', 'Fuel_Deviation']
    rtn = merge(rtn, base_col, dfEcoInfo, added_col)

    for index, row in rtn.iterrows():
        rtn.at[index, 'Fuel_Deviation'] *= np.random.randn(1)
    rtn['Payment'] = rtn['Quantity'] * \
        (rtn['Fuel_Deviation'] + rtn['Fuel_SupplyCost'] + rtn['Fuel_WasteFee'])
    return trn[['SimId', 'TransactionId', 'AgentId', 'Commodity', 'Payment', 'Time']]


del _fcdeps, _fcschema


_dcdeps = [ ('TimeSeriesPower', ('SimId', 'AgentId'), 'Value'),
            ('AgentEntry', ('EnterTime', 'Lifetime', 'AgentId'), 'Spec'),
            ('Info', ('InitialYear', 'InitialMonth'), 'Duration'), ('EconomicInfo', (('Agent', 'AgentId'), ('Decommissioning', 'Duration')), ('Decommissioning', 'OvernightCost'))]

_dcschema = [('SimId', ts.UUID), ('AgentId', ts.INT), ('Payment',
          ts.DOUBLE), ('Time', ts.INT)]

@metric(name='DecommissioningCost', depends=_dcdeps, schema=_dcschema)
def decommissioning_cost(series):
    """The Decommissioning cost metric gives the cash flows at each time step corresponding to the reactors decommissioning.
    """
    if series[0].empty:
        return pd.DataFrame()
    dfPower = series[0].reset_index()
    dfPower = dfPower[dfPower['Value'].apply(lambda x: x > 0)]
    dfEntry = series[1].reset_index()
    dfInfo = series[2].reset_index()
    dfEcoInfo = series[3].reset_index()
    tuples = (('Agent', 'AgentId'), ('Decommissioning', 'Duration'), ('Decommissioning', 'OvernightCost'))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    dfEcoInfo.columns = index
    dfEcoInfo = dfEcoInfo.set_index(('Agent', 'AgentId'))
    simDuration = dfInfo['Duration'].iloc[0]
    dfEntry = dfEntry[dfEntry['Lifetime'].apply(lambda x: x > 0)]
    dfEntry = dfEntry[(dfEntry['EnterTime'] + dfEntry['Lifetime']).apply(lambda x: x < simDuration)] # only reactors that will be decommissioned
    reactorsId = dfEntry[dfEntry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    rtn = pd.DataFrame()
    for id in reactorsId:
        tmp = dfEcoInfo.loc[id].copy()
        if isinstance(tmp,pd.DataFrame):
            tmp = tmp.iloc[0]
        duration = int(tmp.loc[('Decommissioning', 'Duration')])
        overnightCost = tmp.loc[('Decommissioning', 'OvernightCost')]
        cashFlowShape = capital_shape(duration - duration // 2, duration // 2)
        powerCapacity = dfPower[dfPower.AgentId==id]['Value'].iloc[0]
        cashFlow = cashFlowShape * powerCapacity * overnightCost
        entryTime = dfEntry[dfEntry.AgentId==id]['EnterTime'].iloc[0]
        lifetime = dfEntry[dfEntry.AgentId==id]['Lifetime'].iloc[0]
        rtn = pd.concat([rtn,pd.DataFrame({'AgentId': id, 'Time': list(range(lifetime + entryTime, lifetime + entryTime + duration + 1)), 'Payment': cashFlow})], ignore_index=True)
    rtn['SimId'] = dfPower['SimId'].iloc[0]
    subset = rtn.columns.tolist()
    subset = subset[-1:]+subset[:-1]
    rtn = rtn[subset]
    return rtn[rtn['Time'].apply(lambda x: x >= 0 and x < simDuration)]

del _dcdeps, _dcschema


_omdeps = ['TimeSeriesPower', 'EconomicInfo']

_omschema = [('SimId', ts.UUID), ('AgentId', ts.INT), ('Time', ts.INT), 
          ('Payment', ts.DOUBLE)]

@metric(name='OperationMaintenance', depends=_omdeps, schema=_omschema)
def operation_maintenance(s1, s2):
    """The OperationMaintenance metric gives the cash flows at each time step corresponding to the reactor operations and maintenance costs.
    """
    rtn = s1
    dfEcoInfo = s2
    index = ['AgentId', 'FixedCost',
            'VariableCost',
            'Deviation']
    #index = pd.MultiIndex.from_tuples(tuples)
    #dfEcoInfo.columns = index
    dfEcoInfo = dfEcoInfo.set_index('AgentId')
    print(rtn) 
    print(dfEcoInfo)
    dfEcoInfo = dfEcoInfo
    rtn['Payment'] = 0
    rtn['tmp'] = 0
    for id in dfEcoInfo.index:
        if isreactor(rtn, id):
            powerGenerated = rtn[rtn.AgentId==id].loc[:,'Value']
            powerCapacity = max(powerGenerated)
            powerGenerated *= 8760 / 12
            tmp = dfEcoInfo.loc[id].copy()
            if isinstance(tmp,pd.DataFrame):
                tmp = tmp.iloc[0]
            deviation = tmp.loc['Deviation']
            deviation *= np.random.randn(1)
            fixedOM = tmp.loc['FixedCost'] + deviation
            variableOM = tmp.loc['VariableCost'] + deviation
            rtn['tmp'] = powerGenerated * variableOM + powerCapacity * fixedOM
            rtn.loc[:, 'Payment'] += rtn.loc[:, 'tmp'].fillna(0)
    rtn = rtn.reset_index()
    del rtn['Value'], rtn['index'], rtn['tmp']
    return rtn

del _omdeps, _omschema


_eideps = [('AgentEntry', ('AgentId', 'Prototype'), 'ParentId')]

_eischema = [('Agent_Prototype', ts.STRING), ('Agent_AgentId', ts.INT),
        ('Agent_ParentId', ts.INT), ('Finance_ReturnOnDebt', ts.DOUBLE),
        ('Finance_ReturnOnEquity', ts.DOUBLE), ('Finance_TaxRate', ts.DOUBLE),
        ('Finance_DiscountRate', ts.DOUBLE), ('Capital_beforePeak', ts.INT),
        ('Capital_afterPeak', ts.INT), ('Capital_constructionDuration', ts.INT),
        ('Capital_Deviation', ts.DOUBLE), ('Capital_OvernightCost', ts.DOUBLE),
        ('Decommissioning_Duration', ts.INT), 
        ('Decommissioning_OvernightCost', ts.DOUBLE), 
        ('OperationMaintenance_FixedCost', ts.DOUBLE),
        ('OperationMaintenance_VariableCost', ts.DOUBLE),
        ('OperationMaintenance_Deviation', ts.DOUBLE), 
        ('FuelCommodity', ts.STRING), ('Fuel_SupplyCost', ts.DOUBLE), 
        ('Fuel_WasteFee', ts.DOUBLE), ('Fuel_Deviation', ts.DOUBLE), 
        ('Truncation_Begin', ts.INT), ('Truncation_End', ts.INT)]
        
@metric(name='EconomicInfo', depends=_eideps, schema=_eischema)
def economic_info(series):
    """The EconomicInfo metric stores all economic data needed to calculate the economic metrics. These economic parameters are originally written in 'parameters.xml'.
    """
    tuples = ['Agent_Prototype', 'Agent_AgentId', 'Agent_ParentId',
            'Finance_ReturnOnDebt', 'Finance_ReturnOnEquity', 'Finance_TaxRate',
            'Finance_DiscountRate', 'Captial_beforePeak', 'Captial_afterPeak',
            'Captial_constructionDuration', 'Captial_Deviation',
            'Captial_OvernightCost', 'Decommissioning_Duration',
            'Decommissioning_OvernightCost', 'OperationMaintenance_FixedCost',
            'OperationMaintenance_VariableCost',
            'OperationMaintenance_Deviation', 'Fuel_Commodity',
            'Fuel_SupplyCost', 'Fuel_WasteFee', 'Fuel_Deviation',
            'Truncation_Begin', 'Truncation_End']
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    rtn = pd.DataFrame(index=index)
    dfEntry = series[0].reset_index()
    agentIndex = dfEntry.reset_index().set_index('AgentId')['index']
    rtn = rtn.T
    rtn['Agent_Prototype'] = dfEntry['Prototype']
    rtn['Agent_AgentId'] = dfEntry['AgentId']
    rtn['Agent_ParentId'] = dfEntry['ParentId']
    parametersInput = 'parameters.xml'
    tree = ET.parse(parametersInput)
    root = tree.getroot()
    truncation = root.find('truncation')
    rtn['Truncation_Begin'] = int(truncation.find('simulation_begin').text)
    rtn['Truncation_End'] = int(truncation.find('simulation_end').text)
    finance = root.find('finance')
    if not finance == None:
        rtn.loc[:, 'Finance_TaxRate'] = float(finance.find('tax_rate').text)
        rtn.loc[:, 'Finance_ReturnOnDebt'] = float(finance.find('return_on_debt').text)
        rtn.loc[:, 'Finance_ReturnOnEquity'] = float(finance.find('return_on_equity').text)
        rtn.loc[:, 'Finance_DiscountRate'] = float(finance.find('discount_rate').text)
    capital = root.find('capital')
    if not capital == None:
        rtn.loc[:, 'Capital_beforePeak'] = int(capital.find('beforePeak').text)
        rtn.loc[:, 'Capital_afterPeak'] = int(capital.find('afterPeak').text)
        rtn.loc[:, 'Capital_constructionDuration'] = int(capital.find('constructionDuration').text)
        rtn.loc[:, 'Capital_Deviation'] = float(capital.find('deviation').text)
        rtn.loc[:, 'Capital_OvernightCost'] = float(capital.find('overnight_cost').text)
    decommissioning = root.find('decommissioning')
    if not decommissioning == None:
        rtn.loc[:, 'Decommissioning_Duration'] = int(decommissioning.find('duration').text)
        rtn.loc[:, 'Decommissioning_OvernightCost'] = float(decommissioning.find('overnight_cost').text)
    operation_maintenance = root.find('operation_maintenance')
    if not operation_maintenance == None:
        rtn.loc[:, 'OperationMaintenance_FixedCost'] = float(operation_maintenance.find('fixed').text)
        rtn.loc[:, 'OperationMaintenance_VariableCost'] = float(operation_maintenance.find('variable').text)
        rtn.loc[:, 'OperationMaintenance_Deviation'] = float(operation_maintenance.find('deviation').text)
    fuel = root.find('fuel')
    indexCopy = rtn.index.copy()
    if not fuel == None:
        for type in fuel.findall('type'):
            supply = float(type.find('supply_cost').text)
            waste = float(type.find('waste_fee').text)
            name = type.find('name').text
            deviation = float(type.find('deviation').text)
            for j in indexCopy:
                if np.isnan(rtn.loc[j, 'Fuel_SupplyCost']):
                    rtn.loc[j, 'Fuel_Commodity'] = name
                    rtn.loc[j, 'Fuel_SupplyCost'] = supply
                    rtn.loc[j, 'Fuel_WasteFee'] = waste
                    rtn.loc[j, 'Fuel_Deviation'] = deviation
                else:
                    indice = rtn.index.size
                    rtn.loc[indice] = rtn.loc[j]
                    rtn.loc[indice, 'Fuel_Commodity'] = name
                    rtn.loc[indice, 'Fuel_SupplyCost'] = supply
                    rtn.loc[indice, 'Fuel_WasteFee'] = waste
                    rtn.loc[indice, 'Fuel_Deviation'] = deviation
    for region in root.findall('region'):
        idRegion = int(region.find('id').text)
        finance = region.find('finance')
        if not finance == None:
            returnOnDebt = float(finance.find('return_on_debt').text)
            returnOnEquity = float(finance.find('return_on_equity').text)
            taxRate = float(finance.find('tax_rate').text)
            discountRate = float(finance.find('discount_rate').text)
            rtn.loc[agentIndex[idRegion], 'Finance_TaxRate'] = taxRate
            rtn.loc[agentIndex[idRegion], 'Finance_ReturnOnDebt'] = returnOnDebt
            rtn.loc[agentIndex[idRegion], 'Finance_ReturnOnEquity'] = returnOnEquity
            rtn.loc[gent_index[idRegion], 'Finance_DiscountRate'] = discountRate
            for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
                rtn.loc[agentIndex[idInstitution], 'Finance_TaxRate'] = taxRate
                rtn.loc[agentIndex[idInstitution], 'Finance_ReturnOnDebt'] = returnOnDebt
                rtn.loc[agentIndex[idInstitution], 'Finance_ReturnOnEquity'] = returnOnEquity
                rtn.loc[gent_index[idInstitution], 'Finance_DiscountRate'] = discountRate
                for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
                    rtn.loc[agentIndex[idFacility], 'Finance_TaxRate'] = taxRate
                    rtn.loc[agentIndex[idFacility], 'Finance_ReturnOnDebt'] = returnOnDebt
                    rtn.loc[agentIndex[idFacility], 'Finance_ReturnOnEquity'] = returnOnEquity
                    rtn.loc[gent_index[idFacility], 'Finance_DiscountRate'] = discountRate
        capital = region.find('capital')
        if capital is not None:
            beforePeak = int(capital.find('beforePeak').text)
            afterPeak = int(capital.find('afterPeak').text)
            constructionDuration = int(capital.find('constructionDuration').text)
            deviation = float(capital.find('deviation').text)
            overnightCost = float(capital.find('overnight_cost').text)
            rtn.loc[agentIndex[idRegion], 'Captial_beforePeak'] = beforePeak
            rtn.loc[agentIndex[idRegion], 'Captial_afterPeak'] = afterPeak
            rtn.loc[agentIndex[idRegion], 'Captial_constructionDuration'] = constructionDuration
            rtn.loc[agentIndex[idRegion], 'Captial_Deviation'] = deviation
            rtn.loc[agentIndex[idRegion], 'Captial_OvernightCost'] = overnightCost
            for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
                rtn.loc[agentIndex[idInstitution], 'Captial_beforePeak'] = beforePeak
                rtn.loc[agentIndex[idInstitution], 'Captial_afterPeak'] = afterPeak
                rtn.loc[agentIndex[idInstitution], 'Captial_constructionDuration'] = constructionDuration
                rtn.loc[agentIndex[idInstitution], 'Captial_Deviation'] = deviation
                rtn.loc[agentIndex[idInstitution], 'Captial_OvernightCost'] = overnightCost
                for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
                    rtn.loc[agentIndex[idFacility], 'Captial_beforePeak'] = beforePeak
                    rtn.loc[agentIndex[idFacility], 'Captial_afterPeak'] = afterPeak
                    rtn.loc[agentIndex[idFacility], 'Captial_constructionDuration'] = constructionDuration
                    rtn.loc[agentIndex[idFacility], 'Captial_Deviation'] = deviation
                    rtn.loc[agentIndex[idFacility], 'Captial_OvernightCost'] = overnightCost
        decommissioning = region.find('decommissioning')
        if decommissioning is not None:
            duration = int(decommissioning.find('duration').text)
            overnightCost = float(decommissioning.find('overnight_cost').text)
            rtn.loc[agentIndex[idRegion], 'Decommissioning_Duration'] = duration
            rtn.loc[agentIndex[idRegion], 'Decommissioning_OvernightCost'] = overnightCost
            for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
                rtn.loc[agentIndex[idInstitution], 'Decommissioning_Duration'] = duration
                rtn.loc[agentIndex[idInstitution], 'Decommissioning_OvernightCost'] = overnightCost
                for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
                    rtn.loc[agentIndex[idFacility], 'Decommissioning_Duration'] = duration
                    rtn.loc[agentIndex[idFacility], 'Decommissioning_OvernightCost'] = overnightCost
        operation_maintenance = region.find('operation_maintenance')
        if operation_maintenance is not None:
            fixed = float(operation_maintenance.find('fixed').text)
            variable = float(operation_maintenance.find('variable').text)
            deviation = float(operation_maintenance.find('deviation').text)
            rtn.loc[agentIndex[idRegion], 'OperationMaintenance_FixedCost'] = fixed
            rtn.loc[agentIndex[idRegion], 'OperationMaintenance_VariableCost'] = variable
            rtn.loc[agentIndex[idRegion], 'OperationMaintenance_Deviation'] = deviation
            for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
                rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_FixedCost'] = fixed
                rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_VariableCost'] = variable
                rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_Deviation'] = deviation
                for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_FixedCost'] = fixed
                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_VariableCost'] = variable
                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_Deviation'] = deviation
        fuel = region.find('fuel')
        if fuel is not None:
            for type in fuel.findall('type'):
                supply = float(type.find('supply_cost').text)
                waste = float(type.find('waste_fee').text)
                name = type.find('name').text
                deviation = float(type.find('deviation').text)
                if np.isnan(rtn.loc[agentIndex[idRegion], 'Fuel_SupplyCost']):
                    rtn.loc[agentIndex[idRegion], 'Fuel_Commodity'] = name
                    rtn.loc[agentIndex[idRegion], 'Fuel_SupplyCost'] = supply
                    rtn.loc[agentIndex[idRegion], 'Fuel_WasteFee'] = waste
                    rtn.loc[agentIndex[idRegion], 'Fuel_Deviation'] = deviation
                else:
                    indice = rtn.index.size
                    rtn.loc[indice] = rtn.loc[agentIndex[idRegion]]
                    rtn.loc[indice, 'Fuel_Commodity'] = name
                    rtn.loc[indice, 'Fuel_SupplyCost'] = supply
                    rtn.loc[indice, 'Fuel_WasteFee'] = waste
                    rtn.loc[indice, 'Fuel_Deviation'] = deviation
            for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
                if np.isnan(rtn.loc[agentIndex[idInstitution], 'Fuel_SupplyCost']):
                    rtn.loc[agentIndex[idInstitution], 'Fuel_Commodity'] = name
                    rtn.loc[agentIndex[idInstitution], 'Fuel_SupplyCost'] = supply
                    rtn.loc[agentIndex[idInstitution], 'Fuel_WasteFee'] = waste
                    rtn.loc[agentIndex[idInstitution], 'Fuel_Deviation'] = deviation
                else:
                    indice = rtn.index.size
                    rtn.loc[indice] = rtn.loc[agentIndex[idInstitution]]
                    rtn.loc[indice, 'Fuel_Commodity'] = name
                    rtn.loc[indice, 'Fuel_SupplyCost'] = supply
                    rtn.loc[indice, 'Fuel_WasteFee'] = waste
                    rtn.loc[indice, 'Fuel_Deviation'] = deviation
            for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
                if np.isnan(rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost']):
                    rtn.loc[agentIndex[idFacility], 'Fuel_Commodity'] = name
                    rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost'] = supply
                    rtn.loc[agentIndex[idFacility], 'Fuel_WasteFee'] = waste
                    rtn.loc[agentIndex[idFacility], 'Fuel_Deviation'] = deviation
                else:
                    indice = rtn.index.size
                    rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
                    rtn.loc[indice, 'Fuel_Commodity'] = name
                    rtn.loc[indice, 'Fuel_SupplyCost'] = supply
                    rtn.loc[indice, 'Fuel_WasteFee'] = waste
                    rtn.loc[indice, 'Fuel_Deviation'] = deviation
        for institution in region.findall('institution'):
            idInstitution = int(institution.find('id').text)
            finance = institution.find('finance')
            if finance is not None:
                returnOnDebt = float(finance.find('return_on_debt').text)
                returnOnEquity = float(finance.find('return_on_equity').text)
                taxRate = float(finance.find('tax_rate').text)
                discountRate = float(finance.find('discount_rate').text)
                rtn.loc[agentIndex[idInstitution], 'Finance_TaxRate'] = taxRate
                rtn.loc[agentIndex[idInstitution], 'Finance_ReturnOnDebt'] = returnOnDebt
                rtn.loc[agentIndex[idInstitution],'Finance_ReturnOnEquity'] = returnOnEquity
                rtn.loc[gent_index[idInstitution], 'Finance_DiscountRate'] = discountRate
                for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
                    rtn.loc[agentIndex[idFacility], 'Finance_TaxRate'] = taxRate
                    rtn.loc[agentIndex[idFacility], 'Finance_ReturnOnDebt'] = returnOnDebt
                    rtn.loc[agentIndex[idFacility], 'Finance_ReturnOnEquity'] = returnOnEquity
                    rtn.loc[gent_index[idFacility], 'Finance_DiscountRate'] = discountRate
            capital = institution.find('capital')
            if capital is not None:
                beforePeak = int(capital.find('beforePeak').text)
                afterPeak = int(capital.find('afterPeak').text)
                constructionDuration = int(capital.find('constructionDuration').text)
                deviation = float(capital.find('deviation').text)
                overnightCost = float(capital.find('overnight_cost').text)
                rtn.loc[agentIndex[idInstitution], 'Captial_beforePeak'] = beforePeak
                rtn.loc[agentIndex[idInstitution], 'Captial_afterPeak'] = afterPeak
                rtn.loc[agentIndex[idInstitution], 'Captial_constructionDuration'] = constructionDuration
                rtn.loc[agentIndex[idInstitution], 'Captial_Deviation'] = deviation
                rtn.loc[agentIndex[idInstitution], 'Captial_OvernightCost'] = overnightCost
                for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
                    rtn.loc[agentIndex[idFacility], 'Captial_beforePeak'] = beforePeak
                    rtn.loc[agentIndex[idFacility], 'Captial_constructionDuration'] = constructionDuration
                    rtn.loc[agentIndex[idFacility], 'Captial_Deviation'] = deviation
                    rtn.loc[agentIndex[idFacility], 'Captial_OvernightCost'] = overnightCost
            decommissioning = institution.find('decommissioning')
            if decommissioning is not None:
                duration = int(decommissioning.find('duration').text)
                overnightCost = float(decommissioning.find('overnight_cost').text)
                rtn.loc[agentIndex[idInstitution], 'Decommissioning_Duration'] = duration
                rtn.loc[agentIndex[idInstitution], 'Decommissioning_OvernightCost'] = overnightCost
                for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
                    rtn.loc[agentIndex[idFacility], 'Decommissioning_Duration'] = duration
                    rtn.loc[agentIndex[idFacility], 'Decommissioning_OvernightCost'] = overnightCost
            operation_maintenance = institution.find('operation_maintenance')
            if operation_maintenance is not None:
                fixed = float(operation_maintenance.find('fixed').text)
                variable = float(operation_maintenance.find('variable').text)
                deviation = float(operation_maintenance.find('deviation').text)
                rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_FixedCost'] = fixed
                rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_VariableCost'] = variable
                rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_Deviation'] = deviation
                for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_FixedCost'] = fixed
                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_VariableCost'] = variable
                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_Deviation'] = deviation
            fuel = institution.find('fuel')
            if fuel is not None:
                for type in fuel.findall('type'):
                    supply = float(type.find('supply_cost').text)
                    waste = float(type.find('waste_fee').text)
                    name = type.find('name').text
                    deviation = float(type.find('deviation').text)
                    if np.isnan(rtn.loc[agentIndex[idInstitution], 'Fuel_SupplyCost']):
                        rtn.loc[agentIndex[idInstitution], 'Fuel_Commodity'] = name
                        rtn.loc[agentIndex[idInstitution], 'Fuel_SupplyCost'] = supply
                        rtn.loc[agentIndex[idInstitution], 'Fuel_WasteFee'] = waste
                        rtn.loc[agentIndex[idInstitution], 'Fuel_Deviation'] = deviation
                    else:
                        indice = rtn.index.size
                        rtn.loc[indice] = rtn.loc[agentIndex[idInstitution]]
                        rtn.loc[indice, 'Fuel_Commodity'] = name
                        rtn.loc[indice, 'Fuel_SupplyCost'] = supply
                        rtn.loc[indice, 'Fuel_WasteFee'] = waste
                        rtn.loc[indice, 'Fuel_Deviation'] = deviation
                    for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
                        if np.isnan(rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost']):
                            rtn.loc[agentIndex[idFacility], 'Fuel_Commodity'] = name
                            rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost'] = supply
                            rtn.loc[agentIndex[idFacility], 'Fuel_WasteFee'] = waste
                            rtn.loc[agentIndex[idFacility], 'Fuel_Deviation'] = deviation
                        else:
                            indice = rtn.index.size
                            rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
                            rtn.loc[indice, 'Fuel_Commodity'] = name
                            rtn.loc[indice, 'Fuel_SupplyCost'] = supply
                            rtn.loc[indice, 'Fuel_WasteFee'] = waste
                            rtn.loc[indice, 'Fuel_Deviation'] = deviation
            for prototype in institution.findall('prototype'):
                name = prototype.find('name').text
                tmp = dfEntry[dfEntry.ParentId==idInstitution]
                facilityIdList = tmp[tmp.Prototype==name].loc[:,'AgentId'].tolist()
                capital = prototype.find('capital')
                if capital is not None:
                    beforePeak = int(capital.find('beforePeak').text)
                    afterPeak = int(capital.find('afterPeak').text)
                    constructionDuration = int(capital.find('constructionDuration').text)
                    deviation = float(capital.find('deviation').text)
                    overnight = float(capital.find('overnight_cost').text)
                    for idFacility in facilityIdList:
                        rtn.loc[agentIndex[idFacility], 'Captial_beforePeak'] = beforePeak
                        rtn.loc[agentIndex[idFacility], 'Captial_afterPeak'] = afterPeak
                        rtn.loc[agentIndex[idFacility], 'Captial_constructionDuration'] = constructionDuration
                        rtn.loc[agentIndex[idFacility], 'Captial_Deviation'] = deviation
                        rtn.loc[agentIndex[idFacility], 'Captial_OvernightCost'] = overnight
                operation_maintenance = prototype.find('operation_maintenance')
                if operation_maintenance is not None:
                    fixed = float(operation_maintenance.find('fixed').text)
                    variable = float(operation_maintenance.find('variable').text)
                    deviation = float(operation_maintenance.find('deviation').text)
                    for idFacility in facilityIdList:
                        rtn.loc[agentIndex[idFacility], 'OperationMaintenance_FixedCost'] = fixed
                        rtn.loc[agentIndex[idFacility], 'OperationMaintenance_VariableCost'] = variable
                        rtn.loc[agentIndex[idFacility], 'OperationMaintenance_Deviation'] = deviation
                fuel = prototype.find('fuel')
                if fuel is not None:
                    for type in fuel.findall('type'):
                        supply = float(type.find('supply_cost').text)
                        waste = float(type.find('waste_fee').text)
                        name = type.find('name').text
                        deviation = float(type.find('deviation').text)
                        for idFacility in facilityIdList:
                            if np.isnan(rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost']):
                                rtn.loc[agentIndex[idFacility], 'Fuel_Commodity'] = name
                                rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost'] = supply
                                rtn.loc[agentIndex[idFacility], 'Fuel_WasteFee'] = waste
                                rtn.loc[agentIndex[idFacility], 'Fuel_Deviation'] = deviation
                            else:
                                indice = rtn.index.size
                                rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
                                rtn.loc[indice, 'Fuel_Commodity'] = name
                                rtn.loc[indice, 'Fuel_SupplyCost'] = supply
                                rtn.loc[indice, 'Fuel_WasteFee'] = waste
                                rtn.loc[indice, 'Fuel_Deviation'] = deviation
                decommissioning = prototype.find('decommissioning')
                if decommissioning is not None:
                    duration = int(decommissioning.find('duration').text)
                    overnight = float(decommissioning.find('overnight_cost').text)
                    for idFacility in facilityIdList:
                        rtn.loc[agentIndex[idFacility], 'Decommissioning_Duration'] = duration
                        rtn.loc[agentIndex[idFacility], 'Decommissioning_OvernightCost'] = overnight
                for facility in prototype.findall('facility'):
                    idFacility = int(facility.find('id').text)
                    capital = facility.find('capital')
                    if capital is not None:
                        rtn.loc[agentIndex[idFacility], 'Captial_beforePeak'] = int(capital.find('beforePeak').text)
                        rtn.loc[agentIndex[idFacility], 'Captial_afterPeak'] = int(capital.find('afterPeak').text)
                        rtn.loc[agentIndex[idFacility], 'Captial_constructionDuration'] = int(capital.find('constructionDuration').text)
                        rtn.loc[agentIndex[idFacility], 'Captial_Deviation'] = float(capital.find('deviation').text)
                        rtn.loc[agentIndex[idFacility], 'Captial_OvernightCost'] = float(capital.find('overnight_cost').text)
                    operation_maintenance = facility.find('operation_maintenance')
                    if operation_maintenance is not None:
                        rtn.loc[agentIndex[idFacility], 'OperationMaintenance_FixedCost'] = float(operation_maintenance.find('fixed').text)
                        rtn.loc[agentIndex[idFacility], 'OperationMaintenance_VariableCost'] = float(operation_maintenance.find('variable').text)
                        rtn.loc[agentIndex[idFacility], 'OperationMaintenance_Deviation'] = float(operation_maintenance.find('deviation').text)
                    fuel = facility.find('fuel')
                    if fuel is not None:
                        for type in fuel.findall('type'):
                            supply = float(type.find('supply_cost').text)
                            waste = float(type.find('waste_fee').text)
                            name = type.find('name').text
                            deviation = float(type.find('deviation').text)
                            if np.isnan(rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost']):
                                rtn.loc[agentIndex[idFacility], 'Fuel_Commodity'] = name
                                rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost'] = supply
                                rtn.loc[agentIndex[idFacility], 'Fuel_WasteFee'] = waste
                                rtn.loc[agentIndex[idFacility], 'Fuel_Deviation'] = deviation
                            else:
                                indice = rtn.index.size
                                rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
                                rtn.loc[indice, 'Fuel_Commodity'] = name
                                rtn.loc[indice, 'Fuel_SupplyCost'] = supply
                                rtn.loc[indice, 'Fuel_WasteFee'] = waste
                                rtn.loc[indice, 'Fuel_Deviation'] = deviation
                    decommissioning = facility.find('decommissioning')
                    if decommissioning is not None:
                        rtn.loc[agentIndex[idFacility], 'Decommissioning_Duration'] = int(decommissioning.find('duration').text)
                        rtn.loc[agentIndex[idFacility], 'Decommissioning_OvernightCost'] = float(decommissioning.find('overnight_cost').text)
    return rtn
    
del _eideps, _eischema



"""The functions below aim at calculating more complex metrics than the simple cash flows corresponding to the construction, fuel, O&M and decommissioning costs. The metrics can be calculated at an agent, institution, region or simulation level.
"""

        
#######################################
# Metrics derived from the cash flows #
#######################################



# Reactor level

def annual_costs(outputDb, reactorId, capital=True):
    """Input : sqlite output database and reactor's AgentId. It is possible not to take into account the construction costs (capital=False) if the reactor is supposed to have been built before the beginning of the simulation.
    Output : total reactor costs per year over its lifetime.
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo.loc[0, 'Duration']
    initialYear = dfInfo.loc[0, 'InitialYear']
    initialMonth = dfInfo.loc[0, 'InitialMonth']
    dfEntry = evaler.eval('AgentEntry').reset_index()
    commissioning = dfEntry[dfEntry.AgentId==reactorId]['EnterTime'].iloc[0]
    dfCapitalCosts = evaler.eval('CapitalCost').reset_index()
    dfCapitalCosts = dfCapitalCosts[dfCapitalCosts.AgentId==reactorId].copy()
    dfCapitalCosts = dfCapitalCosts.groupby('Time').sum()
    costs = pd.DataFrame({'Capital' : dfCapitalCosts['Payment']}, index=list(range(duration)))
    dfDecommissioningCosts = evaler.eval('DecommissioningCost').reset_index()
    dfDecommissioningCosts = dfDecommissioningCosts[dfDecommissioningCosts.AgentId==reactorId].copy()
    dfDecommissioningCosts = dfDecommissioningCosts.groupby('Time').sum()
    costs['Decommissioning'] = dfDecommissioningCosts['Payment']
    dfOMCosts = evaler.eval('OperationMaintenance').reset_index()
    dfOMCosts = dfOMCosts[dfOMCosts.AgentId==reactorId].copy()
    dfOMCosts = dfOMCosts.groupby('Time').sum()
    costs['OandM'] = dfOMCosts['Payment']
    dfFuelCosts = evaler.eval('FuelCost').reset_index()
    dfFuelCosts = dfFuelCosts[dfFuelCosts.AgentId==reactorId].copy()
    dfFuelCosts = dfFuelCosts.groupby('Time').sum()
    costs['Fuel'] = dfFuelCosts['Payment']
    costs = costs.fillna(0)
    costs['Year'] = (costs.index + initialMonth - 1) // 12 + initialYear
    if not capital:
        del costs['Capital']
    costs = costs.groupby('Year').sum()
    return costs
    
def annual_costs_present_value(outputDb, reactorId, capital=True):
    """Same as annual_cost except all values are actualized to the begin date of the SIMULATION
    """
    costs = annual_costs(outputDb, reactorId, capital)
    actualization = actualization_vector(len(costs))
    actualization.index = costs.index
    return costs.apply(lambda x : x * actualization)
   
def average_cost(outputDb, reactorId, capital=True):
    """Input : sqlite output database, reactor's AgentId
    Output : value (in $/MWh) corresponding to the total costs (sum of annual costs) divided by the total power generated.
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    dfPower = evaler.eval('TimeSeriesPower').reset_index()
    powerGenerated = sum(dfPower[dfPower.AgentId==reactorId].loc[:, 'Value']) * 8760 / 12
    return annual_costs(outputDb, reactorId, capital).sum().sum() / powerGenerated
    
def benefit(outputDb, reactorId):
    """Input : sqlite output database and reactor agent id
    Output : cumulative sum of actualized income and expense (= - expenditures + income)
    """
    costs = - annual_costs(outputDb, reactorId).sum(axis=1)
    powerGenerated = power_generated(outputDb, reactorId) * lcoe(outputDb, reactorId)
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
    return (annualCosts.sum(axis=1) * actualization).fillna(0).sum() / ((powerGenerated * actualization).fillna(0).sum())
        
def period_costs(outputDb, reactorId, t0=0, period=20, capital=True):
    """Input : sqlite output database, reactor id, time window (t0, period) 
    Output : cost at each time step t corresponding to actualized sum of expense in [t+t0, t+t0+period] divided by actualized power generated in [t+t0, t+t0+period]
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
    df = pd.DataFrame(index=list(range(initialYear, initialYear + duration // 12 + 1)))
    df['Power'] = power
    df['Costs'] = costs
    df = df.fillna(0)
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear # year instead of months
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd)))
    rtn['Power'] = pd.Series()
    rtn['Payment'] = pd.Series()
    rtn = rtn.fillna(0)
    for i in range(simulationBegin + t0, simulationBegin + t0 + period):    
        rtn.loc[simulationBegin, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - simulationBegin)
        rtn.loc[simulationBegin, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - simulationBegin)
    for j in range(simulationBegin + 1, simulationEnd):
        rtn.loc[j, 'Power'] = rtn.loc[j - 1, 'Power'] * (1 + default_discount_rate) - df.loc[j -1 + t0, 'Power'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Power'] / (1 + default_discount_rate) ** (period + t0 - 1)
        rtn.loc[j, 'Payment'] = rtn.loc[j - 1, 'Payment'] * (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Costs'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Costs'] / (1 + default_discount_rate) ** (period + t0 - 1)
    rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
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
    df = pd.DataFrame(index=list(range(initialYear, initialYear + duration // 12 + 1)))
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
            rtn.loc[j, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - j)
            rtn.loc[j, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - j)
            #tmp['WasteManagement'][j] = pd.Series()
    rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
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
    dfPower = dfPower[dfPower['AgentId']==reactorId].copy()
    dfPower['Year'] = (dfPower['Time'] + initialMonth - 1) // 12 + initialYear
    dfPower = dfPower.groupby('Year').sum()
    rtn = pd.Series(dfPower['Value'] * 8760 / 12, index=list(range(initialYear, initialYear + (initialMonth + duration) // 12 + 1)))
    return rtn.fillna(0)

# Institution level
    
def institution_annual_costs(outputDb, institutionId, capital=True, truncate=True):
    """Input : sqlite output database and institution's AgentId. It is possible not to take into account the construction costs (capital=False) if the reactors are supposed to have been built before the beginning of the simulation. It is also possible to truncate the simulation results and only have access to cash flows occurring between the two dates (begin and end) specified in 'parameters.xml'. The truncation allows to let reactors decommission after the end of the simulation and thus to take into account cash flows that occur after the end of the simulation for example to calculate the LCOE.
    Output : total reactor costs per year over its lifetime at the institution level.
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
    dfEntry = dfEntry[dfEntry.ParentId==institutionId]
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
    dfPower = evaler.eval('TimeSeriesPower')
    reactorIds = dfEntry[dfEntry['AgentId'].apply(lambda x: isreactor(dfPower, x))]['AgentId'].tolist()
    dfCapitalCosts = evaler.eval('CapitalCost').reset_index()
    dfCapitalCosts = dfCapitalCosts[dfCapitalCosts['AgentId'].apply(lambda x: x in reactorIds)]
    dfCapitalCosts = dfCapitalCosts.groupby('Time').sum()
    costs = pd.DataFrame({'Capital' : dfCapitalCosts['Payment']}, index=list(range(0, duration)))
    dfDecommissioningCosts = evaler.eval('DecommissioningCost').reset_index()
    if not dfDecommissioningCosts.empty:
        dfDecommissioningCosts = dfDecommissioningCosts[dfDecommissioningCosts['AgentId'].apply(lambda x: x in reactorIds)]
        dfDecommissioningCosts = dfDecommissioningCosts.groupby('Time').sum()
        costs['Decommissioning'] = dfDecommissioningCosts['Payment']
    dfOMCosts = evaler.eval('OperationMaintenance').reset_index()
    dfOMCosts = dfOMCosts[dfOMCosts['AgentId'].apply(lambda x: x in reactorIds)]
    dfOMCosts = dfOMCosts.groupby('Time').sum()
    costs['OandM'] = dfOMCosts.loc[:, 'Payment']
    dfFuelCosts = evaler.eval('FuelCost').reset_index()
    dfFuelCosts = dfFuelCosts[dfFuelCosts['AgentId'].apply(lambda x: x in reactorIds)]
    dfFuelCosts = dfFuelCosts.groupby('Time').sum()
    costs['Fuel'] = dfFuelCosts.loc[:, 'Payment']
    costs = costs.fillna(0)
    costs['Year'] = (costs.index + initialMonth - 1) // 12 + initialYear
    if truncate:
        endYear = (simulationEnd + initialMonth - 1) // 12 + initialYear
        costs = costs[costs['Year'].apply(lambda x : x <= endYear)]
        beginYear = (simulationBegin + initialMonth - 1) // 12 + initialYear
        costs = costs[costs['Year'].apply(lambda x : x >= beginYear)]
    if not capital:
        del costs['Capital']
    costs = costs.groupby('Year').sum()
    return costs
    
def institution_annual_costs_present_value(outputDb, reactorId, capital=True):
    """Same as annual_cost except all values are actualized to the begin date of the SIMULATION
    """
    costs = institution_annual_costs(outputDb, institutionId, capital)
    actualization = actualization_vector(len(costs))
    actualization.index = costs.index
    return costs.apply(lambda x : x * actualization)
    
def institution_benefit(outputDb, institutionId):
    """Input : sqlite output database and institution agent id
    Output : cumulative sum of income and expense (= - expenditures + income)
    """
    costs = - institution_annual_costs(outputDb, institutionId).sum(axis=1)
    power_gen = institution_power_generated(outputDb, institutionId) * institution_average_lcoe(outputDb, institutionId)['Average LCOE']
    rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
    rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
    actualization = actualization_vector(len(rtn))
    actualization.index = rtn.index
    rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
    return rtn
        
def institution_period_costs(outputDb, institutionId, t0=0, period=20, capital=True):
    """Input : sqlite output database, institution id, time window (t0, period) 
    Output : cost at each time step t corresponding to actualized sum of expense in [t+t0, t+t0+period] divided by actualized power generated in [t+t0, t+t0+period]
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
    costs = institution_annual_costs(outputDb, institutionId, capital, truncate=False)
    costs = costs.sum(axis=1)
    power = institution_power_generated(outputDb, institutionId, truncate=False)
    df = pd.DataFrame(index=list(range(initialYear, initialYear + duration // 12 + 1)))
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
        rtn.loc[simulationBegin, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - simulationBegin)
        rtn.loc[simulationBegin, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - simulationBegin)
    for j in range(simulationBegin + 1, simulationEnd):
        rtn.loc[j, 'Power'] = rtn.loc[j - 1, 'Power'] * (1 + default_discount_rate) - df.loc[j -1 + t0, 'Power'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Power'] / (1 + default_discount_rate) ** (period + t0 - 1)
        rtn.loc[j, 'Payment'] = rtn.loc[j - 1, 'Payment'] * (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Costs'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Costs'] / (1 + default_discount_rate) ** (period + t0 - 1)
    rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
    actualization = actualization_vector(len(rtn))
    actualization.index = rtn.index
    rtn['Actualized'] = rtn['Ratio'] * actualization
    return rtn
    
def institution_period_costs2(outputDb, institutionId, t0=0, period=20, capital=True):
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
    costs = institution_annual_costs(outputDb, institutionId, capital, truncate=False)
    costs = costs.sum(axis=1)
    power = institution_power_generated(outputDb, institutionId, truncate=False)
    df = pd.DataFrame(index=list(range(initialYear, initialYear + duration // 12 + 1)))
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
            rtn.loc[j, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - j)
            rtn.loc[j, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - j)
    rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
    return rtn
        
def institution_power_generated(outputDb, institutionId, truncate=True):
    """Input : sqlite output database and institution agent id
    Output : Sum of electricity generated in MWh every year in the institution reactor fleet
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
    dfEntry = dfEntry[dfEntry.ParentId==institutionId]
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
    dfPower = evaler.eval('TimeSeriesPower')
    reactorIds = dfEntry[dfEntry['AgentId'].apply(lambda x: isreactor(dfPower, x))]['AgentId'].tolist()
    dfPower = evaler.eval('TimeSeriesPower').reset_index()
    dfPower = dfPower[dfPower['AgentId'].apply(lambda x: x in reactorIds)]
    dfPower['Year'] = (dfPower['Time'] + initialMonth - 1) // 12 + initialYear
    dfPower = dfPower.groupby('Year').sum()
    rtn = pd.Series(dfPower['Value'] * 8760 / 12, index=list(range(initialYear, initialYear + (initialMonth + duration) // 12 + 1)))
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
    return (annualCosts.sum(axis=1) * actualization).fillna(0).sum() / ((powerGenerated * actualization).fillna(0).sum())
        

def institution_average_lcoe(outputDb, institutionId):
    """Input : sqlite output database and institution agent id
    Output : Variable cost corresponding at each time step (i.e. every year) to the weighted average of the reactors Levelized Cost of Electricity ($/MWh). A reactor is taken into account at a time step t only if it is active (i.e. already commissioned and not yet decommissioned) at this time step.
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
    dfEntry = dfEntry[dfEntry.ParentId==institutionId]
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
    dfPower = evaler.eval('TimeSeriesPower')
    reactorIds = dfEntry[dfEntry['AgentId'].apply(lambda x: isreactor(dfPower, x))]['AgentId'].tolist()
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
        commissioning = dfEntry[dfEntry.AgentId==id]['EnterTime'].iloc[0]
        lifetime = dfEntry[dfEntry.AgentId==id]['Lifetime'].iloc[0]
        decommissioning = (commissioning + lifetime + initialMonth - 1) // 12 + initialYear
        commissioning = (commissioning + initialMonth - 1) // 12 + initialYear
        power = dfPower[dfPower.AgentId==id]['Value'].iloc[0]
        rtn['Temp'] = pd.Series(tmp, index=list(range(commissioning, decommissioning + 1))) * power
        rtn['Weighted sum'] += rtn['Temp'].fillna(0)
        rtn['Temp2'] = pd.Series(power, index=list(range(commissioning, decommissioning + 1))).fillna(0)
        rtn['Power'] += rtn['Temp2'].fillna(0)
    rtn['Average LCOE'] = rtn['Weighted sum'] / rtn['Power']
    return rtn.fillna(0)
        
# Region level

def region_annual_costs(outputDb, regionId, capital=True, truncate=True):
    """Input : sqlite output database and region's AgentId. It is possible not to take into account the construction costs (capital=False) if the reactors are supposed to have been built before the beginning of the simulation. It is also possible to truncate the simulation results and only have access to cash flows occurring between the two dates (begin and end) specified in 'parameters.xml'. The truncation allows to let reactors decommission after the end of the simulation and thus to take into account cash flows that occur after the end of the simulation for example to calculate the LCOE.
    Output : total reactor costs per year over its lifetime at the region level.
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
    tmp = dfEntry[dfEntry.ParentId==regionId]
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
    institutionIds = tmp[tmp.Kind=='Inst']['AgentId'].tolist()
    reactorIds = []
    for id in institutionIds:
        dfEntry2 = dfEntry[dfEntry.ParentId==id]
        reactorIds += dfEntry2[dfEntry2['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    dfCapitalCosts = evaler.eval('CapitalCost').reset_index()
    dfCapitalCosts = dfCapitalCosts[dfCapitalCosts['AgentId'].apply(lambda x: x in reactorIds)]
    dfCapitalCosts = dfCapitalCosts.groupby('Time').sum()
    costs = pd.DataFrame({'Capital' : dfCapitalCosts['Payment']}, index=list(range(duration)))
    dfDecommissioningCosts = evaler.eval('DecommissioningCost').reset_index()
    if not dfDecommissioningCosts.empty:
        dfDecommissioningCosts = dfDecommissioningCosts[dfDecommissioningCosts['AgentId'].apply(lambda x: x in reactorIds)]
        dfDecommissioningCosts = dfDecommissioningCosts.groupby('Time').sum()
        costs['Decommissioning'] = dfDecommissioningCosts['Payment']
    dfOMCosts = evaler.eval('OperationMaintenance').reset_index()
    dfOMCosts = dfOMCosts[dfOMCosts['AgentId'].apply(lambda x: x in reactorIds)]
    dfOMCosts = dfOMCosts.groupby('Time').sum()
    costs['OandM'] = dfOMCosts['Payment']
    dfFuelCosts = evaler.eval('FuelCost').reset_index()
    dfFuelCosts = dfFuelCosts[dfFuelCosts['AgentId'].apply(lambda x: x in reactorIds)]
    dfFuelCosts = dfFuelCosts.groupby('Time').sum()
    costs['Fuel'] = dfFuelCosts['Payment']
    costs = costs.fillna(0)
    costs['Year'] = (costs.index + initialMonth - 1) // 12 + initialYear
    if truncate:
        endYear = (simulationEnd + initialMonth - 1) // 12 + initialYear
        costs = costs[costs['Year'].apply(lambda x : x <= endYear)]
        beginYear = (simulationBegin + initialMonth - 1) // 12 + initialYear
        costs = costs[costs['Year'].apply(lambda x : x >= beginYear)]
    if not capital:
        del costs['Capital']
    costs = costs.groupby('Year').sum()
    return costs
        
def region_annual_costs_present_value(outputDb, regionId, capital=True, truncate=True):
    """Same as annual_cost except all values are actualized to the begin date of the SIMULATION
    """
    costs = region_annual_costs(outputDb, regionId, capital)
    actualization = actualization_vector(len(costs))
    actualization.index = costs.index
    return costs.apply(lambda x : x * actualization)
        
def region_benefit(outputDb, regionId):
    """Input : sqlite output database and region agent id
    Output : cumulative sum of actualized income and expense (= - expenditures + income)
    """
    costs = - region_annual_costs(outputDb, regionId).sum(axis=1)
    power_gen = region_power_generated(outputDb, regionId) * region_average_lcoe(outputDb, regionId)['Average LCOE']
    rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
    rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
    actualization = actualization_vector(len(rtn))
    actualization.index = rtn.index
    rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
    return rtn
        
def region_period_costs(outputDb, regionId, t0=0, period=20, capital=True):
    """Input : sqlite output database, region id, time window (t0, period) 
    Output : cost at each time step t corresponding to actualized sum of expense in [t+t0, t+t0+period] divided by actualized power generated in [t+t0, t+t0+period]
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
    df = pd.DataFrame(index=list(range(initialYear, initialYear + duration // 12 + 1)))
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
        rtn.loc[simulationBegin, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - simulationBegin)
        rtn.loc[simulationBegin, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - simulationBegin)
    for j in range(simulationBegin + 1, simulationEnd + 1):
        rtn.loc[j, 'Power'] = rtn.loc[j - 1, 'Power'] * (1 + default_discount_rate) - df.loc[j -1 + t0, 'Power'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Power'] / (1 + default_discount_rate) ** (period + t0 - 1)
        rtn.loc[j, 'Payment'] = rtn.loc[j - 1, 'Payment'] * (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Costs'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Costs'] / (1 + default_discount_rate) ** (period + t0 - 1)
    rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
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
    df = pd.DataFrame(index=list(range(initialYear, initialYear + duration // 12 + 1)))
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
            rtn.loc[j, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - j)
            rtn.loc[j, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - j)
    rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
    return rtn
        
def region_power_generated(outputDb, regionId, truncate=True):
    """Input : sqlite output database and region agent id
    Output : Sum of electricity generated in MWh every years in the region reactor fleet
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
    tmp = dfEntry[dfEntry.ParentId==regionId]
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
    institutionIds = tmp[tmp.Kind=='Inst']['AgentId'].tolist()
    reactorIds = []
    for id in institutionIds:
        dfEntry2 = dfEntry[dfEntry.ParentId==id]
        reactorIds += dfEntry2[dfEntry2['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    dfPower = evaler.eval('TimeSeriesPower').reset_index()
    dfPower = dfPower[dfPower['AgentId'].apply(lambda x: x in reactorIds)]
    dfPower['Year'] = (dfPower['Time'] + initialMonth - 1) // 12 + initialYear
    dfPower = dfPower.groupby('Year').sum()
    rtn = pd.Series(dfPower['Value'] * 8760 / 12, index=list(range(initialYear, initialYear + (initialMonth + duration) // 12 + 1)))
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
    return (annualCosts.sum(axis=1) * actualization).fillna(0).sum() / ((powerGenerated * actualization).fillna(0).sum())

def region_average_lcoe(outputDb, regionId):
    """Input : sqlite output database and region agent id
    Output : Variable cost corresponding at each time step (i.e. every year) to the weighted average of the reactors Levelized Cost of Electricity ($/MWh). A reactor is taken into account at a time step t if and only if it is active (i.e. already commissioned and not yet decommissioned) at this time step.
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
    tmp = dfEntry[dfEntry.ParentId==regionId]
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
    institutionsId = tmp[tmp.Kind=='Inst']['AgentId'].tolist()
    reactorIds = []
    dfPower = evaler.eval('TimeSeriesPower')
    for id in institutionsId:
        dfEntry2 = dfEntry[dfEntry.ParentId==id]
        reactorIds += dfEntry2[dfEntry2['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear
    simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
    rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
    rtn['Weighted sum'] = 0
    rtn['Power'] = 0
    rtn['Temp'] = pd.Series()
    rtn['Temp2'] = pd.Series()
    for id in reactorIds:
        tmp = lcoe(outputDb, id)
        commissioning = dfEntry[dfEntry.AgentId==id]['EnterTime'].iloc[0]
        lifetime = dfEntry[dfEntry.AgentId==id]['Lifetime'].iloc[0]
        decommissioning = (commissioning + lifetime + initialMonth - 1) // 12 + initialYear
        commissioning = (commissioning + initialMonth - 1) // 12 + initialYear
        power = dfPower[dfPower.AgentId==id]['Value'].iloc[0]
        rtn['Temp'] = pd.Series(tmp, index=list(range(commissioning, decommissioning + 1))) * power
        rtn['Weighted sum'] += rtn['Temp'].fillna(0)
        rtn['Temp2'] = pd.Series(power, index=list(range(commissioning, decommissioning + 1))).fillna(0)
        rtn['Power'] += rtn['Temp2']
    rtn['Average LCOE'] = rtn['Weighted sum'] / rtn['Power']
    return rtn.fillna(0)
    
# Simulation level

def simulation_annual_costs(outputDb, capital=True, truncate=True):
    """Input : sqlite output database. It is possible not to take into account the construction costs (capital=False) if the reactors are supposed to have been built before the beginning of the simulation. It is also possible to truncate the simulation results and only have access to cash flows occurring between the two dates (begin and end) specified in 'parameters.xml'. The truncation allows to let reactors decommission after the end of the simulation and thus to take into account cash flows that occur after the end of the simulation for example to calculate the LCOE.
    Output : total reactor costs per year over its lifetime at the simulation level.
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
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
    reactorIds = dfEntry[dfEntry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    dfCapitalCosts = evaler.eval('CapitalCost').reset_index()
    dfCapitalCosts = dfCapitalCosts[dfCapitalCosts['AgentId'].apply(lambda x: x in reactorIds)]
    mini = min(dfCapitalCosts['Time'])
    dfCapitalCosts = dfCapitalCosts.groupby('Time').sum()
    costs = pd.DataFrame({'Capital' : dfCapitalCosts['Payment']}, index=list(range(0, duration)))
    dfDecommissioningCosts = evaler.eval('DecommissioningCost').reset_index()
    if not dfDecommissioningCosts.empty:
        dfDecommissioningCosts = dfDecommissioningCosts[dfDecommissioningCosts['AgentId'].apply(lambda x: x in reactorIds)]
        dfDecommissioningCosts = dfDecommissioningCosts.groupby('Time').sum()
        costs['Decommissioning'] = dfDecommissioningCosts['Payment']
    dfOMCosts = evaler.eval('OperationMaintenance').reset_index()
    dfOMCosts = dfOMCosts[dfOMCosts['AgentId'].apply(lambda x: x in reactorIds)]
    dfOMCosts = dfOMCosts.groupby('Time').sum()
    costs['OandM'] = dfOMCosts['Payment']
    dfFuelCosts = evaler.eval('FuelCost').reset_index()
    dfFuelCosts = dfFuelCosts[dfFuelCosts['AgentId'].apply(lambda x: x in reactorIds)]
    dfFuelCosts = dfFuelCosts.groupby('Time').sum()
    costs['Fuel'] = dfFuelCosts['Payment']
    costs = costs.fillna(0)
    costs['Year'] = (costs.index + initialMonth - 1) // 12 + initialYear
    if truncate:
        endYear = (simulationEnd + initialMonth - 1) // 12 + initialYear
        costs = costs[costs['Year'].apply(lambda x : x <= endYear)]
        beginYear = (simulationBegin + initialMonth - 1) // 12 + initialYear
        costs = costs[costs['Year'].apply(lambda x : x >= beginYear)]
    if not capital:
        del costs['Capital']
    costs = costs.groupby('Year').sum()
    return costs
        
def simulation_annual_costs_present_value(outputDb, capital=True, truncate=True):
    """Same as annual_cost except all values are actualized to the begin date of the SIMULATION
    """
    df = simulation_annual_costs(outputDb, capital, truncate)
    for year in df.index:
        df.loc[year, :] = df.loc[year, :] / (1 + default_discount_rate) ** (year - df.index[0])
    return df
    
def simulation_benefit(outputDb):
    """Input : sqlite output database
    Output : cumulative sum of total income and total expense (= - expenditures + income) when all reactors of the simulation are taken into account
    """
    costs = - simulation_annual_costs(outputDb).sum(axis=1)
    power_gen = simulation_power_generated(outputDb) * simulation_average_lcoe(outputDb)['Average LCOE']
    rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
    rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
    actualization = actualization_vector(len(rtn))
    actualization.index = rtn.index
    rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
    return rtn
        
def simulation_period_costs(outputDb, t0=0, period=20, capital=True):
    """Input : sqlite output database, time window (t0, period) 
    Output : cost at each time step t corresponding to actualized sum of total expense in [t+t0, t+t0+period] divided by total actualized power generated in [t+t0, t+t0+period] when all reactors of the simulation are taken into account
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
    costs = simulation_annual_costs(outputDb, capital, truncate=False).sum(axis=1)
    power = simulation_power_generated(outputDb, truncate=False)
    df = pd.DataFrame(index=list(range(initialYear, initialYear + duration // 12 + 1)))
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
        rtn.loc[simulationBegin, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - simulationBegin)
        rtn.loc[simulationBegin, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - simulationBegin)
    for j in range(simulationBegin + 1, simulationEnd):
        rtn.loc[j, 'Power'] = rtn.loc[j - 1, 'Power'] * (1 + default_discount_rate) - df.loc[j -1 + t0, 'Power'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Power'] / (1 + default_discount_rate) ** (period + t0 - 1)
        rtn.loc[j, 'Payment'] = rtn.loc[j - 1, 'Payment'] * (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Costs'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Costs'] / (1 + default_discount_rate) ** (period + t0 - 1)
    rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
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
    costs = simulation_annual_costs(outputDb, capital, truncate=False).sum(axis=1)
    power = simulation_power_generated(outputDb, truncate=False)
    df = pd.DataFrame(index=list(range(initialYear, initialYear + duration // 12 + 1)))
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
            rtn.loc[j, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - j)
            rtn.loc[j, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - j)
    rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
    return rtn
        
def simulation_power_generated(outputDb, truncate=True):
    """Input : sqlite output database
    Output : Electricity generated in MWh every years by all the reactors of the simulation
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
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
    reactorIds = dfEntry[dfEntry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    dfPower = evaler.eval('TimeSeriesPower').reset_index()
    dfPower = dfPower[dfPower['AgentId'].apply(lambda x: x in reactorIds)]
    dfPower['Year'] = (dfPower['Time'] + initialMonth - 1) // 12 + initialYear
    dfPower = dfPower.groupby('Year').sum()
    rtn = pd.Series(dfPower['Value'] * 8760 / 12, index=list(range(initialYear, initialYear + (initialMonth + duration) // 12 + 1)))
    rtn.name = 'Power in MWh'
    return rtn.fillna(0)

def simulation_lcoe(outputDb):
    """Input : sqlite output database
    Output : Value corresponding to Levelized Cost of Electricity ($/MWh) when taking into account all reactors commissioned in the simulation
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
    return (annualCosts.sum(axis=1) * actualization).fillna(0).sum() / ((powerGenerated * actualization).fillna(0).sum())

def simulation_average_lcoe(outputDb):
    """Input : sqlite output database and region agent id
    Output : Variable cost corresponding at each time step (i.e. every year) to the weighted average of the reactors Levelized Cost of Electricity ($/MWh). A reactor is taken into account at a time step t if and only if it is in activity (i.e. already commissioned and not yet decommissioned) at this time step.
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
    dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
    reactorIds = dfEntry[dfEntry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
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
        commissioning = dfEntry[dfEntry.AgentId==id]['EnterTime'].iloc[0]
        lifetime = dfEntry[dfEntry.AgentId==id]['Lifetime'].iloc[0]
        decommissioning = (commissioning + lifetime + initialMonth - 1) // 12 + initialYear
        commissioning = (commissioning + initialMonth - 1) // 12 + initialYear
        power = dfPower[dfPower.AgentId==id]['Value'].iloc[0]
        rtn['Temp'] = pd.Series(tmp, index=list(range(commissioning, decommissioning + 1))) * power
        rtn['Weighted sum'] += rtn['Temp'].fillna(0)
        rtn['Temp2'] = pd.Series(power, index=list(range(commissioning, decommissioning + 1)))
        rtn['Power'] += rtn['Temp2'].fillna(0)
        print(id) # test
    rtn['Average LCOE'] = rtn['Weighted sum'] / rtn['Power']
    return rtn.fillna(0)    
