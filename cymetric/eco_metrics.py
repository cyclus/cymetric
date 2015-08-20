#####economic metrics for nuclear power plants#####

from __future__ import print_function, unicode_literals

import inspect

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import math
import os
import matplotlib.pyplot as plt
import warnings

from cymetric.tools import dbopen
from cymetric.evaluator import Evaluator
from cymetric.eco_inputs import actualization_vector, isreactor

try:
    from cymetric.metrics import metric
    from cymetric import cyclus
    from cymetric import schemas
    from cymetric import typesystem as ts
    from cymetric import tools
    from cymetric.evaluator import register_metric
    from cymetric.eco_tools import capital_shape, rapid_cap_begin, rapid_cap_duration, slow_cap_begin, slow_cap_duration, default_cap_begin, default_cap_duration, default_cap_overnight, default_cap_shape, default_discount_rate, isreactor
except ImportError:
    # some wacky CI paths prevent absolute importing, try relative
    from .metrics import metric
    from . import cyclus
    from . import schemas
    from . import typesystem as ts
    from . import tools
    from .evaluator import register_metric
    from .eco_tools import capital_shape, rapid_cap_begin, rapid_cap_duration, slow_cap_begin, slow_cap_duration, default_cap_begin, default_cap_duration, default_cap_overnight, default_cap_shape, default_discount_rate, isreactor

xml_inputs = 'parameters.xml' # This xml file has to be created to store the economic data needed to calculate the EconomicInfo metric

## The actual metrics ##


_ccdeps = [('TimeSeriesPower', ('SimId', 'AgentId', 'Value'), 'Time'), ('AgentEntry', ('AgentId', 'ParentId', 'Spec'), 'EnterTime'), ('Info', ('InitialYear', 'InitialMonth'), 'Duration'), ('EconomicInfo', (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Capital', 'beforePeak'), ('Capital', 'afterPeak'), ('Capital', 'constructionDuration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost')), ('Finance','DiscountRate'))]

_ccschema = [('SimId', ts.UUID), ('AgentId', ts.INT),
             ('Time', ts.INT), ('Payment', ts.DOUBLE)]

@metric(name='CapitalCost', depends=_ccdeps, schema=_ccschema)
def capital_cost(series):
    """The CapitalCost metric gives the cash flows at each time step related to the reactor constructions.
    """
    dfPower = series[0].reset_index()
    dfEntry = series[1].reset_index()
    dfInfo = series[2].reset_index()
    dfEcoInfo = series[3].reset_index()
    tuples = (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Capital', 'beforePeak'), ('Capital', 'afterPeak'), ('Capital', 'constructionDuration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost'), ('Finance','DiscountRate'))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    dfEcoInfo.columns = index
    dfEcoInfo = dfEcoInfo.set_index(('Agent', 'AgentId'))
    simDuration = dfInfo['Duration'].iloc[0]
    #std=3.507*12
    #var=std**2
    dfEntry = pd.DataFrame([dfEntry.EnterTime, dfEntry.AgentId]).transpose()
    dfEntry = dfEntry.set_index(['AgentId'])
    agentIds = dfEcoInfo.index
    rtn = pd.DataFrame()
    for id in agentIds:
    	tmp = dfEcoInfo.loc[id].copy()
    	if isinstance(tmp, pd.DataFrame):
    		tmp = tmp.iloc[0]
    	if isreactor(dfPower, id):
    		deviation = dfEcoInfo.loc[id, ('Capital', 'Deviation')]
    		deviation = deviation * np.random.randn(1)
    		deviation = int(np.around(deviation))
    		beforePeak = int(tmp.loc[('Capital', 'beforePeak')] + deviation)
    		afterPeak = int(tmp.loc[('Capital', 'beforePeak')])
    		constructionDuration = int(tmp.loc[('Capital', 'constructionDuration')] + deviation)
    		overnightCost = tmp.loc[('Capital', 'OvernightCost')]
    		cashFlowShape = capital_shape(beforePeak, afterPeak)
    		powerCapacity = max(dfPower[dfPower.AgentId==id]['Value'])
    		discountRate = tmp.loc[('Finance','DiscountRate')]
    		cashFlow = np.around(cashFlowShape * overnightCost * powerCapacity, 3)
    		cashFlow *= ((1 + discountRate) ** math.ceil((beforePeak + afterPeak) / 12) - 1) / (discountRate * math.ceil((beforePeak + afterPeak) / 12))
    		tmp = pd.DataFrame({'AgentId': id, 'Time': pd.Series(list(range(beforePeak + afterPeak + 1))) + dfEntry.EnterTime[id] - constructionDuration, 'Payment' : cashFlow})
    		rtn = pd.concat([rtn, tmp], ignore_index=True)
    rtn['SimId'] = dfPower['SimId'].iloc[0]
    subset = rtn.columns.tolist()
    subset = subset[3:] + subset[:1] + subset[2:3] + subset[1:2]
    rtn = rtn[subset]
    rtn = rtn[rtn['Time'].apply(lambda x: x >= 0 and x < simDuration)]
    rtn = rtn.reset_index()
    del rtn['index']
    return rtn

del _ccdeps, _ccschema


_fcdeps = [('Resources', ('SimId', 'ResourceId'), 'Quantity'), ('Transactions',
        ('SimId', 'TransactionId', 'ReceiverId', 'ResourceId', 'Commodity'), 
        'Time'), ('EconomicInfo', (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Fuel', 'Commodity'), ('Fuel', 'SupplyCost'), ('Fuel', 'WasteFee'), ('Fuel', 'Deviation')), ('Finance','DiscountRate'))]

_fcschema = [('SimId', ts.UUID), ('TransactionId', ts.INT), ('AgentId', 
          ts.INT), ('Commodity', ts.STRING), ('Payment', ts.DOUBLE), ('Time', 
          ts.INT)]

@metric(name='FuelCost', depends=_fcdeps, schema=_fcschema)
def fuel_cost(series):
    """The FuelCost metric gives the cash flows at each time step related to the reactor fuel costs. It is the sum of the cost of the fuel the reactor have to purchase and the waste fee.
    """
    # fuel_price = 2360 # $/kg
    # see http://www.world-nuclear.org/info/Economic-Aspects/Economics-of-Nuclear-Power/
    dfResources = series[0].reset_index().set_index(['ResourceId'])
    dfTransactions = series[1].reset_index().set_index(['ResourceId'])
    dfEcoInfo = series[2].reset_index()
    tuples = (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Fuel', 'Commodity'), ('Fuel', 'SupplyCost'), ('Fuel', 'WasteFee'), ('Fuel', 'Deviation'), ('Finance','DiscountRate'))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    dfEcoInfo.columns = index
    dfEcoInfo = dfEcoInfo.set_index(('Agent', 'AgentId'))
    dfTransactions['Quantity'] = dfResources.loc[:, 'Quantity']
    dfTransactions['Payment'] = pd.Series()
    dfTransactions.loc[:, 'Payment'] = dfTransactions.loc[:, 'Payment'].fillna(0)
    dfTransactions['Tmp'] = pd.Series()
    for agentId in dfEcoInfo.index:
    	tmpTrans = dfTransactions[dfTransactions.ReceiverId==agentId]
    	if isinstance(dfEcoInfo.loc[agentId, ('Fuel', 'Commodity')], str):
    		commod = dfEcoInfo.loc[agentId, ('Fuel', 'Commodity')]
    		deviation = dfEcoInfo.loc[agentId, ('Fuel', 'Deviation')]
    		tmpTrans2 = tmpTrans[tmpTrans.Commodity==commod]
    		deviation = deviation * np.random.randn(1)
    		price = deviation + dfEcoInfo[dfEcoInfo[('Fuel', 'Commodity')]==commod].loc[agentId, ('Fuel', 'SupplyCost')]
    		dfTransactions.loc[:, 'Tmp'] = tmpTrans2.loc[:, 'Quantity'] * price
    	elif isinstance(dfEcoInfo.loc[agentId, ('Fuel', 'Commodity')], pd.Series):
    		for commod in dfEcoInfo.loc[agentId, ('Fuel', 'Commodity')]:
    			deviation = dfEcoInfo.loc[id, ('Fuel', 'Deviation')]
    			deviation = deviation * np.random.randn(1)
    			price = deviation + dfEcoInfo[dfEcoInfo[('Fuel', 'Commodity')]==commod].loc[agentId, ('Fuel', 'SupplyCost')]
    			tmpTrans2 = tmpTrans[tmpTrans.Commodity==commod]
    			dfTransactions.loc[:, 'Tmp'] = tmpTrans2.loc[:, 'Quantity'] * price		
    	dfTransactions.loc[:, 'Payment'] += dfTransactions.loc[:, 'Tmp'].fillna(0)
    del dfTransactions['Quantity']
    del dfTransactions['Tmp']
    rtn = dfTransactions.reset_index()
    subset = rtn.columns.tolist()
    subset = subset[1:5]+subset[6:]+subset[5:6]
    rtn = rtn[subset]
    rtn.columns = ['SimId', 'TransactionId', 'AgentId', 'Commodity', 'Payment', 'Time']
    return rtn

del _fcdeps, _fcschema


_dcdeps = [ ('TimeSeriesPower', ('SimId', 'AgentId'), 'Value'),
			('AgentEntry', ('EnterTime', 'Lifetime', 'AgentId'), 'Spec'),
			('Info', ('InitialYear', 'InitialMonth'), 'Duration'), ('EconomicInfo', (('Agent', 'AgentId'), ('Decommissioning', 'Duration')), ('Decommissioning', 'OvernightCost'))]

_dcschema = [('SimId', ts.UUID), ('AgentId', ts.INT), ('Payment',
          ts.DOUBLE), ('Time', ts.INT)]

@metric(name='DecommissioningCost', depends=_dcdeps, schema=_dcschema)
def decommissioning_cost(series):
    """The Decommissioning metric gives the cash flows at each time step related to the reactor decommissioning.
    """
    # cost = 750000 # decommission cost in $/MW d'Haeseler
    # duration = 150 # decommission lasts about 15 yrs
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


_omdeps = [('TimeSeriesPower', ('SimId', 'AgentId', 'Time'), 'Value'), ('EconomicInfo', (('Agent', 'AgentId'), ('OperationMaintenance', 'FixedCost'), ('OperationMaintenance', 'VariableCost')), ('OperationMaintenance', 'Deviation'))]

_omschema = [('SimId', ts.UUID), ('AgentId', ts.INT), ('Time', ts.INT), 
          ('Payment', ts.DOUBLE)]

@metric(name='OperationMaintenance', depends=_omdeps, schema=_omschema)
def operation_maintenance(series):
    """The OperationMaintenance metric gives the cash flows at each time step related to the reactor operations and maintenance costs.
    """
    #cost = 10 # $/MWh
    rtn = series[0].reset_index()
    dfEcoInfo = series[1].reset_index()
    tuples = (('Agent', 'AgentId'), ('OperationMaintenance', 'FixedCost'), ('OperationMaintenance', 'VariableCost'), ('OperationMaintenance', 'Deviation'))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    dfEcoInfo.columns = index
    dfEcoInfo = dfEcoInfo.set_index(('Agent', 'AgentId'))
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
    		deviation = tmp.loc[('OperationMaintenance', 'Deviation')]
    		deviation *= np.random.randn(1)
    		fixedOM = tmp.loc[('OperationMaintenance', 'FixedCost')] + deviation
    		variableOM = tmp.loc[('OperationMaintenance', 'VariableCost')] + deviation
    		rtn['tmp'] = powerGenerated * variableOM + powerCapacity * fixedOM
    		rtn.loc[:, 'Payment'] += rtn.loc[:, 'tmp'].fillna(0)
    rtn = rtn.reset_index()
    del rtn['Value'], rtn['index'], rtn['tmp']
    return rtn

del _omdeps, _omschema


_eideps = [('AgentEntry', ('AgentId', 'Prototype'), 'ParentId')]

_eischema = [('AgentId', ts.INT), ('Prototype', ts.STRING), ('ParentId', ts.INT), ('BeginMonth', ts.INT), ('EndMonth', ts.INT), ('DiscountRate', ts.DOUBLE)]
		
@metric(name='EconomicInfo', depends=_eideps, schema=_eischema)
def economic_info(series):
    """The EconomicInfo metric stores all economic data needed to calculate the economic metrics. These economic parameters are originally written in 'parameters.xml'.
    """
    tuples = [('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Agent', 'ParentId'), ('Finance','ReturnOnDebt'), ('Finance','ReturnOnEquity'), ('Finance','TaxRate'), ('Finance','DiscountRate'), ('Capital', 'beforePeak'), ('Capital', 'afterPeak'), ('Capital', 'constructionDuration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost'), ('Decommissioning', 'Duration'), ('Decommissioning', 'OvernightCost'), ('OperationMaintenance', 'FixedCost'), ('OperationMaintenance', 'VariableCost'), ('OperationMaintenance', 'Deviation'), ('Fuel', 'Commodity'), ('Fuel', 'SupplyCost'), ('Fuel', 'WasteFee'), ('Fuel', 'Deviation'), ('Truncation', 'Begin'), ('Truncation', 'End')]
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    rtn = pd.DataFrame(index=index)
    dfEntry = series[0].reset_index()
    agentIndex = dfEntry.reset_index().set_index('AgentId')['index']
    rtn = rtn.T
    rtn[('Agent', 'Prototype')] = dfEntry['Prototype']
    rtn[('Agent', 'AgentId')] = dfEntry['AgentId']
    rtn[('Agent', 'ParentId')] = dfEntry['ParentId']
    parametersInput = 'parameters.xml'
    tree = ET.parse(parametersInput)
    root = tree.getroot()
    truncation = root.find('truncation')
    rtn[('Truncation', 'Begin')] = int(truncation.find('simulation_begin').text)
    rtn[('Truncation', 'End')] = int(truncation.find('simulation_end').text)
    finance = root.find('finance')
    if not finance == None:
    	rtn.loc[:, ('Finance', 'TaxRate')] = float(finance.find('tax_rate').text)
    	rtn.loc[:, ('Finance', 'ReturnOnDebt')] = float(finance.find('return_on_debt').text)
    	rtn.loc[:, ('Finance', 'ReturnOnEquity')] = float(finance.find('return_on_equity').text)
    	rtn.loc[:, ('Finance', 'DiscountRate')] = float(finance.find('discount_rate').text)
    capital = root.find('capital')
    if not capital == None:
    	rtn.loc[:, ('Capital', 'beforePeak')] = int(capital.find('beforePeak').text)
    	rtn.loc[:, ('Capital', 'afterPeak')] = int(capital.find('afterPeak').text)
    	rtn.loc[:, ('Capital', 'constructionDuration')] = int(capital.find('constructionDuration').text)
    	rtn.loc[:, ('Capital', 'Deviation')] = float(capital.find('deviation').text)
    	rtn.loc[:, ('Capital', 'OvernightCost')] = float(capital.find('overnight_cost').text)
    decommissioning = root.find('decommissioning')
    if not decommissioning == None:
    	rtn.loc[:, ('Decommissioning', 'Duration')] = int(decommissioning.find('duration').text)
    	rtn.loc[:, ('Decommissioning', 'OvernightCost')] = float(decommissioning.find('overnight_cost').text)
    operation_maintenance = root.find('operation_maintenance')
    if not operation_maintenance == None:
    	rtn.loc[:, ('OperationMaintenance', 'FixedCost')] = float(operation_maintenance.find('fixed').text)
    	rtn.loc[:, ('OperationMaintenance', 'VariableCost')] = float(operation_maintenance.find('variable').text)
    	rtn.loc[:, ('OperationMaintenance', 'Deviation')] = float(operation_maintenance.find('deviation').text)
    fuel = root.find('fuel')
    indexCopy = rtn.index.copy()
    if not fuel == None:
    	for type in fuel.findall('type'):
    		supply = float(type.find('supply_cost').text)
    		waste = float(type.find('waste_fee').text)
    		name = type.find('name').text
    		deviation = float(type.find('deviation').text)
    		for j in indexCopy:
    			if np.isnan(rtn.loc[j, ('Fuel', 'SupplyCost')]):
    				rtn.loc[j, ('Fuel', 'Commodity')] = name
    				rtn.loc[j, ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[j, ('Fuel', 'WasteFee')] = waste
    				rtn.loc[j, ('Fuel', 'Deviation')] = deviation
    			else:
    				indice = rtn.index.size
    				rtn.loc[indice] = rtn.loc[j]
    				rtn.loc[indice, ('Fuel', 'Commodity')] = name
    				rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    				rtn.loc[indice, ('Fuel', 'Deviation')] = deviation
    # discount rate is only possible at sim or reg level
    for region in root.findall('region'):
    	idRegion = int(region.find('id').text)
    	finance = region.find('finance')
    	if not finance == None:
    		returnOnDebt = float(finance.find('return_on_debt').text)
    		returnOnEquity = float(finance.find('return_on_equity').text)
    		taxRate = float(finance.find('tax_rate').text)
    		discountRate = float(finance.find('discount_rate').text)
    		rtn.loc[agentIndex[idRegion], ('Finance', 'TaxRate')] = taxRate
    		rtn.loc[agentIndex[idRegion], ('Finance','ReturnOnDebt')] = returnOnDebt
    		rtn.loc[agentIndex[idRegion],('Finance','ReturnOnEquity')] = returnOnEquity
    		rtn.loc[gent_index[idRegion], ('Finance','DiscountRate')] = discountRate
    		for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    			rtn.loc[agentIndex[idInstitution], ('Finance', 'TaxRate')] = taxRate
    			rtn.loc[agentIndex[idInstitution], ('Finance','ReturnOnDebt')] = returnOnDebt
    			rtn.loc[agentIndex[idInstitution], ('Finance','ReturnOnEquity')] = returnOnEquity
    			rtn.loc[gent_index[idInstitution], ('Finance','DiscountRate')] = discountRate
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('Finance', 'TaxRate')] = taxRate
    				rtn.loc[agentIndex[idFacility], ('Finance','ReturnOnDebt')] = returnOnDebt
    				rtn.loc[agentIndex[idFacility], ('Finance','ReturnOnEquity')] = returnOnEquity
    				rtn.loc[gent_index[idFacility], ('Finance','DiscountRate')] = discountRate
    	capital = region.find('capital')
    	if capital is not None:
    		beforePeak = int(capital.find('beforePeak').text)
    		afterPeak = int(capital.find('afterPeak').text)
    		constructionDuration = int(capital.find('constructionDuration').text)
    		deviation = float(capital.find('deviation').text)
    		overnightCost = float(capital.find('overnight_cost').text)
    		rtn.loc[agentIndex[idRegion], ('Capital', 'beforePeak')] = beforePeak
    		rtn.loc[agentIndex[idRegion], ('Capital', 'afterPeak')] = afterPeak
    		rtn.loc[agentIndex[idRegion], ('Capital', 'constructionDuration')] = constructionDuration
    		rtn.loc[agentIndex[idRegion], ('Capital', 'Deviation')] = deviation
    		rtn.loc[agentIndex[idRegion], ('Capital', 'OvernightCost')] = overnightCost
    		for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'beforePeak')] = beforePeak
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'afterPeak')] = afterPeak
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'constructionDuration')] = constructionDuration
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Deviation')] = deviation
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'OvernightCost')] = overnightCost
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('Capital', 'beforePeak')] = beforePeak
    				rtn.loc[agentIndex[idFacility], ('Capital', 'afterPeak')] = afterPeak
    				rtn.loc[agentIndex[idFacility], ('Capital', 'constructionDuration')] = constructionDuration
    				rtn.loc[agentIndex[idFacility], ('Capital', 'Deviation')] = deviation
    				rtn.loc[agentIndex[idFacility], ('Capital', 'OvernightCost')] = overnightCost
    	decommissioning = region.find('decommissioning')
    	if decommissioning is not None:
    		duration = int(decommissioning.find('duration').text)
    		overnightCost = float(decommissioning.find('overnight_cost').text)
    		rtn.loc[agentIndex[idRegion], ('Decommissioning', 'Duration')] = duration
    		rtn.loc[agentIndex[idRegion], ('Decommissioning', 'OvernightCost')] = overnightCost
    		for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    			rtn.loc[agentIndex[idInstitution], ('Decommissioning', 'Duration')] = duration
    			rtn.loc[agentIndex[idInstitution], ('Decommissioning', 'OvernightCost')] = overnightCost
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('Decommissioning', 'Duration')] = duration
    				rtn.loc[agentIndex[idFacility], ('Decommissioning', 'OvernightCost')] = overnightCost
    	operation_maintenance = region.find('operation_maintenance')
    	if operation_maintenance is not None:
    		fixed = float(operation_maintenance.find('fixed').text)
    		variable = float(operation_maintenance.find('variable').text)
    		deviation = float(operation_maintenance.find('deviation').text)
    		rtn.loc[agentIndex[idRegion], ('OperationMaintenance', 'FixedCost')] = fixed
    		rtn.loc[agentIndex[idRegion], ('OperationMaintenance', 'VariableCost')] = variable
    		rtn.loc[agentIndex[idRegion], ('OperationMaintenance', 'Deviation')] = deviation
    		for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'FixedCost')] = fixed
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'VariableCost')] = variable
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'Deviation')] = deviation
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'FixedCost')] = fixed
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'VariableCost')] = variable
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'Deviation')] = deviation
    	fuel = region.find('fuel')
    	if fuel is not None:
    		for type in fuel.findall('type'):
    			supply = float(type.find('supply_cost').text)
    			waste = float(type.find('waste_fee').text)
    			name = type.find('name').text
    			deviation = float(type.find('deviation').text)
    			if np.isnan(rtn.loc[agentIndex[idRegion], ('Fuel', 'SupplyCost')]):
    				rtn.loc[agentIndex[idRegion], ('Fuel', 'Commodity')] = name
    				rtn.loc[agentIndex[idRegion], ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[agentIndex[idRegion], ('Fuel', 'WasteFee')] = waste
    				rtn.loc[agentIndex[idRegion], ('Fuel', 'Deviation')] = deviation
    			else:
    				indice = rtn.index.size
    				rtn.loc[indice] = rtn.loc[agentIndex[idRegion]]
    				rtn.loc[indice, ('Fuel', 'Commodity')] = name
    				rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    				rtn.loc[indice, ('Fuel', 'Deviation')] = deviation
    		for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    			if np.isnan(rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')]):
    				rtn.loc[agentIndex[idInstitution], ('Fuel', 'Commodity')] = name
    				rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[agentIndex[idInstitution], ('Fuel', 'WasteFee')] = waste
    				rtn.loc[agentIndex[idInstitution], ('Fuel', 'Deviation')] = deviation
    			else:
    				indice = rtn.index.size
    				rtn.loc[indice] = rtn.loc[agentIndex[idInstitution]]
    				rtn.loc[indice, ('Fuel', 'Commodity')] = name
    				rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    				rtn.loc[indice, ('Fuel', 'Deviation')] = deviation
    		for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    			if np.isnan(rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')]):
    				rtn.loc[agentIndex[idFacility], ('Fuel', 'Commodity')] = name
    				rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[agentIndex[idFacility], ('Fuel', 'WasteFee')] = waste
    				rtn.loc[agentIndex[idFacility], ('Fuel', 'Deviation')] = deviation
    			else:
    				indice = rtn.index.size
    				rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    				rtn.loc[indice, ('Fuel', 'Commodity')] = name
    				rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    				rtn.loc[indice, ('Fuel', 'Deviation')] = deviation
    	for institution in region.findall('institution'):
    		idInstitution = int(institution.find('id').text)
    		finance = institution.find('finance')
    		if finance is not None:
    			returnOnDebt = float(finance.find('return_on_debt').text)
    			returnOnEquity = float(finance.find('return_on_equity').text)
    			taxRate = float(finance.find('tax_rate').text)
    			discountRate = float(finance.find('discount_rate').text)
    			rtn.loc[agentIndex[idInstitution], ('Finance', 'TaxRate')] = taxRate
    			rtn.loc[agentIndex[idInstitution], ('Finance','ReturnOnDebt')] = returnOnDebt
    			rtn.loc[agentIndex[idInstitution],('Finance','ReturnOnEquity')] = returnOnEquity
    			rtn.loc[gent_index[idInstitution], ('Finance','DiscountRate')] = discountRate
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('Finance', 'TaxRate')] = taxRate
    				rtn.loc[agentIndex[idFacility], ('Finance','ReturnOnDebt')] = returnOnDebt
    				rtn.loc[agentIndex[idFacility], ('Finance','ReturnOnEquity')] = returnOnEquity
    				rtn.loc[gent_index[idFacility], ('Finance','DiscountRate')] = discountRate
    		capital = institution.find('capital')
    		if capital is not None:
    			beforePeak = int(capital.find('beforePeak').text)
    			afterPeak = int(capital.find('afterPeak').text)
    			constructionDuration = int(capital.find('constructionDuration').text)
    			deviation = float(capital.find('deviation').text)
    			overnightCost = float(capital.find('overnight_cost').text)
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'beforePeak')] = beforePeak
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'afterPeak')] = afterPeak
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'constructionDuration')] = constructionDuration
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Deviation')] = deviation
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'OvernightCost')] = overnightCost
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('Capital', 'beforePeak')] = beforePeak
    				rtn.loc[agentIndex[idFacility], ('Capital', 'constructionDuration')] = constructionDuration
    				rtn.loc[agentIndex[idFacility], ('Capital', 'Deviation')] = deviation
    				rtn.loc[agentIndex[idFacility], ('Capital', 'OvernightCost')] = overnightCost
    		decommissioning = institution.find('decommissioning')
    		if decommissioning is not None:
    			duration = int(decommissioning.find('duration').text)
    			overnightCost = float(decommissioning.find('overnight_cost').text)
    			rtn.loc[agentIndex[idInstitution], ('Decommissioning', 'Duration')] = duration
    			rtn.loc[agentIndex[idInstitution], ('Decommissioning', 'OvernightCost')] = overnightCost
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('Decommissioning', 'Duration')] = duration
    				rtn.loc[agentIndex[idFacility], ('Decommissioning', 'OvernightCost')] = overnightCost
    		operation_maintenance = institution.find('operation_maintenance')
    		if operation_maintenance is not None:
    			fixed = float(operation_maintenance.find('fixed').text)
    			variable = float(operation_maintenance.find('variable').text)
    			deviation = float(operation_maintenance.find('deviation').text)
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'FixedCost')] = fixed
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'VariableCost')] = variable
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'Deviation')] = deviation
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'FixedCost')] = fixed
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'VariableCost')] = variable
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'Deviation')] = deviation
    		fuel = institution.find('fuel')
    		if fuel is not None:
    			for type in fuel.findall('type'):
    				supply = float(type.find('supply_cost').text)
    				waste = float(type.find('waste_fee').text)
    				name = type.find('name').text
    				deviation = float(type.find('deviation').text)
    				if np.isnan(rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')]):
    					rtn.loc[agentIndex[idInstitution], ('Fuel', 'Commodity')] = name
    					rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')] = supply
    					rtn.loc[agentIndex[idInstitution], ('Fuel', 'WasteFee')] = waste
    					rtn.loc[agentIndex[idInstitution], ('Fuel', 'Deviation')] = deviation
    				else:
    					indice = rtn.index.size
    					rtn.loc[indice] = rtn.loc[agentIndex[idInstitution]]
    					rtn.loc[indice, ('Fuel', 'Commodity')] = name
    					rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    					rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    					rtn.loc[indice, ('Fuel', 'Deviation')] = deviation
    				for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    					if np.isnan(rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')]):
    						rtn.loc[agentIndex[idFacility], ('Fuel', 'Commodity')] = name
    						rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')] = supply
    						rtn.loc[agentIndex[idFacility], ('Fuel', 'WasteFee')] = waste
    						rtn.loc[agentIndex[idFacility], ('Fuel', 'Deviation')] = deviation
    					else:
    						indice = rtn.index.size
    						rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    						rtn.loc[indice, ('Fuel', 'Commodity')] = name
    						rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    						rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    						rtn.loc[indice, ('Fuel', 'Deviation')] = deviation
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
    					rtn.loc[agentIndex[idFacility], ('Capital', 'beforePeak')] = beforePeak
    					rtn.loc[agentIndex[idFacility], ('Capital', 'afterPeak')] = afterPeak
    					rtn.loc[agentIndex[idFacility], ('Capital', 'constructionDuration')] = constructionDuration
    					rtn.loc[agentIndex[idFacility], ('Capital', 'Deviation')] = deviation
    					rtn.loc[agentIndex[idFacility], ('Capital', 'OvernightCost')] = overnight
    			operation_maintenance = prototype.find('operation_maintenance')
    			if operation_maintenance is not None:
    				fixed = float(operation_maintenance.find('fixed').text)
    				variable = float(operation_maintenance.find('variable').text)
    				deviation = float(operation_maintenance.find('deviation').text)
    				for idFacility in facilityIdList:
    					rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'FixedCost')] = fixed
    					rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'VariableCost')] = variable
    					rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'Deviation')] = deviation
    			fuel = prototype.find('fuel')
    			if fuel is not None:
    				for type in fuel.findall('type'):
    					supply = float(type.find('supply_cost').text)
    					waste = float(type.find('waste_fee').text)
    					name = type.find('name').text
    					deviation = float(type.find('deviation').text)
    					for idFacility in facilityIdList:
    						if np.isnan(rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')]):
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'Commodity')] = name
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')] = supply
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'WasteFee')] = waste
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'Deviation')] = deviation
    						else:
    							indice = rtn.index.size
    							rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    							rtn.loc[indice, ('Fuel', 'Commodity')] = name
    							rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    							rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    							rtn.loc[indice, ('Fuel', 'Deviation')] = deviation
    			decommissioning = prototype.find('decommissioning')
    			if decommissioning is not None:
    				duration = int(decommissioning.find('duration').text)
    				overnight = float(decommissioning.find('overnight_cost').text)
    				for idFacility in facilityIdList:
    					rtn.loc[agentIndex[idFacility], ('Decommissioning', 'Duration')] = duration
    					rtn.loc[agentIndex[idFacility], ('Decommissioning', 'OvernightCost')] = overnight
    			for facility in prototype.findall('facility'):
    				idFacility = int(facility.find('id').text)
    				capital = facility.find('capital')
    				if capital is not None:
    					rtn.loc[agentIndex[idFacility], ('Capital', 'beforePeak')] = int(capital.find('beforePeak').text)
    					rtn.loc[agentIndex[idFacility], ('Capital', 'afterPeak')] = int(capital.find('afterPeak').text)
    					rtn.loc[agentIndex[idFacility], ('Capital', 'constructionDuration')] = int(capital.find('constructionDuration').text)
    					rtn.loc[agentIndex[idFacility], ('Capital', 'Deviation')] = float(capital.find('deviation').text)
    					rtn.loc[agentIndex[idFacility], ('Capital', 'OvernightCost')] = float(capital.find('overnight_cost').text)
    				operation_maintenance = facility.find('operation_maintenance')
    				if operation_maintenance is not None:
    					rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'FixedCost')] = float(operation_maintenance.find('fixed').text)
    					rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'VariableCost')] = float(operation_maintenance.find('variable').text)
    					rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'Deviation')] = float(operation_maintenance.find('deviation').text)
    				fuel = facility.find('fuel')
    				if fuel is not None:
    					for type in fuel.findall('type'):
    						supply = float(type.find('supply_cost').text)
    						waste = float(type.find('waste_fee').text)
    						name = type.find('name').text
    						deviation = float(type.find('deviation').text)
    						if np.isnan(rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')]):
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'Commodity')] = name
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')] = supply
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'WasteFee')] = waste
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'Deviation')] = deviation
    						else:
    							indice = rtn.index.size
    							rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    							rtn.loc[indice, ('Fuel', 'Commodity')] = name
    							rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    							rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    							rtn.loc[indice, ('Fuel', 'Deviation')] = deviation
    				decommissioning = facility.find('decommissioning')
    				if decommissioning is not None:
    					rtn.loc[agentIndex[idFacility], ('Decommissioning', 'Duration')] = int(decommissioning.find('duration').text)
    					rtn.loc[agentIndex[idFacility], ('Decommissioning', 'OvernightCost')] = float(decommissioning.find('overnight_cost').text)
    return rtn
	
del _eideps, _eischema



"""Functions to calculate more complex metrics that those calculated in eco_metrics.py. Some function are also dedicated to visualization of economic calculations (plotting of metrics). All metrics can be calculated at an agent, institution, region or simulation level.
"""

        
#######################################
# Metrics derived from the cash flows #
#######################################

# Reactor level

def annual_costs(outputDb, reactorId, capital=True):
    """Input : sqlite output database and reactor's AgentId. It is possible to not take into account the construction costs (capital=False) if the reactor is supposed to have been built before the beginning of the simulation.
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
	Output : cumulative sum of income and expense (= - expenditures + income)
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
			#tmp['WasteManagement'][j] = pd.Series()
	rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = rtn['Ratio'] * actualization
	return rtn
	
def period_costs2(outputDb, reactorId, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
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
	Output : Electricity generated in MWh every years
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
			#tmp['WasteManagement'][j] = pd.Series()
	rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = rtn['Ratio'] * actualization
	return rtn
	
def institution_period_costs2(outputDb, institutionId, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
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
	simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear # year instead of months
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
		
def institution_power_generated(outputDb, institutionId, truncate=True):
	"""Input : sqlite output database and institution agent id
	Output : Sum of electricity generated in MWh every years in the institution reactor fleet
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
	Output : Variable cost corresponding at each time step (i.e. every year) to the weighted average of the reactors Levelized Cost of Electricity ($/MWh). A reactor is taken into account at time step t only if it is in activity (i.e. already commissioned and not yet decommissioned) at time step t.
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
	simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear # year instead of months
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
	Output : cumulative sum of income and expense (= - expenditures + income)
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
	simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear # year instead of months
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
			#tmp['WasteManagement'][j] = pd.Series()
	rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = rtn['Ratio'] * actualization
	return rtn
	
def region_period_costs2(outputDb, regionId, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
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
	simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear # year instead of months
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
	Output : Variable cost corresponding at each time step (i.e. every year) to the weighted average of the reactors Levelized Cost of Electricity ($/MWh). A reactor is taken into account at time step t if and only if it is in activity (i.e. already commissioned and not yet decommissioned) at time step t.
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
	simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear # year instead of months
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
	
def simulation_benefitl(outputDb):
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
			#tmp['WasteManagement'][j] = pd.Series()
	rtn['Ratio'] = rtn['Payment'] / rtn ['Power'] * (rtn['Power'] > 1)
	return rtn
	
def simulation_period_costs2(outputDb, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
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
	simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear # year instead of months
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
	Output : Variable cost corresponding at each time step (i.e. every year) to the weighted average of the reactors Levelized Cost of Electricity ($/MWh). A reactor is taken into account at time step t if and only if it is in activity (i.e. already commissioned and not yet decommissioned) at time step t.
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
	simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear # year instead of months
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