#####economic metrics for nuclear power plants#####

from __future__ import print_function, unicode_literals

import inspect

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import math
import os


try:
    from cymetric.metrics import metric
    from cymetric import cyclus
    from cymetric import schemas
    from cymetric import typesystem as ts
    from cymetric import tools
    from cymetric.evaluator import register_metric
    from cymetric.eco_inputs import capital_shape, rapid_cap_begin, rapid_cap_duration, slow_cap_begin, slow_cap_duration, default_cap_begin, default_cap_duration, default_cap_overnight, default_cap_shape, default_discount_rate, isreactor
except ImportError:
    # some wacky CI paths prevent absolute importing, try relative
    from .metrics import metric
    from . import cyclus
    from . import schemas
    from . import typesystem as ts
    from . import tools
    from .evaluator import register_metric
    from .eco_inputs import capital_shape, rapid_cap_begin, rapid_cap_duration, slow_cap_begin, slow_cap_duration, default_cap_begin, default_cap_duration, default_cap_overnight, default_cap_shape, default_discount_rate, isreactor

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
    tuples = (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Capital', 'beforePeak'), ('Capital', 'constructionDuration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost'), ('Finance','DiscountRate'))
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
    		variance = deviation ** 2
    		deviation = np.random.poisson(variance) - variance
    		beforePeak = int(tmp.loc[('Capital', 'beforePeak')] + deviation)
    		afterPeak = int(tmp.loc[('Capital', 'beforePeak')])
    		constructionDuration = int(tmp.loc[('Capital', 'constructionDuration')] + deviation)
    		overnightCost = tmp.loc[('Capital', 'OvernightCost')]
    		cashFlowShape = capital_shape(beforePeak, afterPeak)
    		powerCapacity = max(dfPower[dfPower.AgentId==id]['Value'])
    		discountRate = tmp.loc[('Finance','DiscountRate')]
    		cashFlow = np.around(cashFlowShape * overnightCost * powerCapacity, 3)
    		cashFlow *= ((1 + discountRate) ** math.ceil(duration / 12) - 1) / (discountRate * math.ceil(duration / 12))
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
        'Time'), ('EconomicInfo', (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Fuel', 'SupplyCost'), ('Fuel', 'WasteFee')), ('Finance','DiscountRate'))]

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
    tuples = (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Fuel', 'SupplyCost'), ('Fuel', 'WasteFee'), ('Finance','DiscountRate'))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    dfEcoInfo.columns = index
    dfEcoInfo = dfEcoInfo.set_index(('Agent', 'AgentId'))
    dfTransactions['Quantity'] = dfResources.loc[:, 'Quantity']
    dfTransactions['Payment'] = pd.Series()
    dfTransactions.loc[:, 'Payment'] = dfTransactions.loc[:, 'Payment'].fillna(0)
    dfTransactions['Tmp'] = pd.Series()
    for agentId in dfEcoInfo.index:
    	tmpEcoInfo = dfEcoInfo.loc[agentId].copy()
    	tmpTrans = dfTransactions[dfTransactions.ReceiverId==agentId]
    	for commod in dfEcoInfo.loc[agentId, ('Fuel', 'Commodity')]:
    		price = dfEcoInfo[dfEcoInfo.Commodity==commod].loc[agentId, ('Fuel', 'SupplyCost')]
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
    	cashFlowShape = capital_shape(duration // 2, duration // 2 + 1)
    	powerCapacity = dfPower[dfPower.AgentId==id]['Value'].iloc[0]
    	cashFlow = cashFlowShape * powerCapacity * overnightCost
    	entryTime = dfEntry[dfEntry.AgentId==id]['EnterTime'].iloc[0]
    	lifetime = dfEntry[dfEntry.AgentId==id]['Lifetime'].iloc[0]
    	rtn = pd.concat([rtn,pd.DataFrame({'AgentId': id, 'Time': list(range(lifetime + entryTime, lifetime + entryTime + duration)), 'Payment': cashFlow})], ignore_index=True)
    rtn['SimId'] = dfPower['SimId'].iloc[0]
    subset = rtn.columns.tolist()
    subset = subset[-1:]+subset[:-1]
    rtn = rtn[subset]
    return rtn[rtn['Time'].apply(lambda x: x >= 0 and x < simDuration)]

del _dcdeps, _dcschema


_omdeps = [('TimeSeriesPower', ('SimId', 'AgentId', 'Time'), 'Value'), ('EconomicInfo', (('Agent', 'AgentId'), ('OperationMaintenance', 'FixedCost')), ('OperationMaintenance', 'VariableCost'))]

_omschema = [('SimId', ts.UUID), ('AgentId', ts.INT), ('Time', ts.INT), 
          ('Payment', ts.DOUBLE)]

@metric(name='OperationMaintenance', depends=_omdeps, schema=_omschema)
def operation_maintenance(series):
    """The OperationMaintenance metric gives the cash flows at each time step related to the reactor operations and maintenance costs.
    """
    #cost = 10 # $/MWh
    rtn = series[0].reset_index()
    dfEcoInfo = series[1].reset_index()
    tuples = (('Agent', 'AgentId'), ('OperationMaintenance', 'FixedCost'), ('OperationMaintenance', 'VariableCost'))
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
    		fixedOM = tmp.loc[('OperationMaintenance', 'FixedCost')]
    		variableOM = tmp.loc[('OperationMaintenance', 'VariableCost')]
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
    tuples = [('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Agent', 'ParentId'), ('Finance','ReturnOnDebt'), ('Finance','ReturnOnEquity'), ('Finance','TaxRate'), ('Finance','DiscountRate'), ('Capital', 'beforePeak'), ('Capital', 'afterPeak'), ('Capital', 'constructionDuration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost'), ('Decommissioning', 'Duration'), ('Decommissioning', 'OvernightCost'), ('OperationMaintenance', 'FixedCost'), ('OperationMaintenance', 'VariableCost'), ('Fuel', 'Commodity'), ('Fuel', 'SupplyCost'), ('Fuel', 'WasteFee'), ('Truncation', 'Begin'), ('Truncation', 'End')]
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
    fuel = root.find('fuel')
    indexCopy = rtn.index.copy()
    if not fuel == None:
    	for type in fuel.findall('type'):
    		supply = float(type.find('supply_cost').text)
    		waste = float(type.find('waste_fee').text)
    		name = type.find('name').text
    		for j in indexCopy:
    			if np.isnan(rtn.loc[j, ('Fuel', 'SupplyCost')]):
    				rtn.loc[j, ('Fuel', 'Commodity')] = name
    				rtn.loc[j, ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[j, ('Fuel', 'WasteFee')] = waste
    			else:
    				indice = rtn.index.size
    				rtn.loc[indice] = rtn.loc[j]
    				rtn.loc[indice, ('Fuel', 'Commodity')] = name
    				rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
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
    		rtn.loc[agentIndex[idRegion], ('OperationMaintenance', 'FixedCost')] = fixed
    		rtn.loc[agentIndex[idRegion], ('OperationMaintenance', 'VariableCost')] = variable
    		for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'FixedCost')] = fixed
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'VariableCost')] = variable
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'FixedCost')] = fixed
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'VariableCost')] = variable
    	fuel = region.find('fuel')
    	if fuel is not None:
    		for type in fuel.findall('type'):
    			supply = float(type.find('supply_cost').text)
    			waste = float(type.find('waste_fee').text)
    			name = type.find('name').text
    			if np.isnan(rtn.loc[agentIndex[idRegion], ('Fuel', 'SupplyCost')]):
    				rtn.loc[agentIndex[idRegion], ('Fuel', 'Commodity')] = name
    				rtn.loc[agentIndex[idRegion], ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[agentIndex[idRegion], ('Fuel', 'WasteFee')] = waste
    			else:
    				indice = rtn.index.size
    				rtn.loc[indice] = rtn.loc[agentIndex[idRegion]]
    				rtn.loc[indice, ('Fuel', 'Commodity')] = name
    				rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    		for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    			if np.isnan(rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')]):
    				rtn.loc[agentIndex[idInstitution], ('Fuel', 'Commodity')] = name
    				rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[agentIndex[idInstitution], ('Fuel', 'WasteFee')] = waste
    			else:
    				indice = rtn.index.size
    				rtn.loc[indice] = rtn.loc[agentIndex[idInstitution]]
    				rtn.loc[indice, ('Fuel', 'Commodity')] = name
    				rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    		for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    			if np.isnan(rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')]):
    				rtn.loc[agentIndex[idFacility], ('Fuel', 'Commodity')] = name
    				rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[agentIndex[idFacility], ('Fuel', 'WasteFee')] = waste
    			else:
    				indice = rtn.index.size
    				rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    				rtn.loc[indice, ('Fuel', 'Commodity')] = name
    				rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
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
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'FixedCost')] = fixed
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'VariableCost')] = variable
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'FixedCost')] = fixed
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'VariableCost')] = variable
    		fuel = institution.find('fuel')
    		if fuel is not None:
    			for type in fuel.findall('type'):
    				supply = float(type.find('supply_cost').text)
    				waste = float(type.find('waste_fee').text)
    				name = type.find('name').text
    				if np.isnan(rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')]):
    					rtn.loc[agentIndex[idInstitution], ('Fuel', 'Commodity')] = name
    					rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')] = supply
    					rtn.loc[agentIndex[idInstitution], ('Fuel', 'WasteFee')] = waste
    				else:
    					indice = rtn.index.size
    					rtn.loc[indice] = rtn.loc[agentIndex[idInstitution]]
    					rtn.loc[indice, ('Fuel', 'Commodity')] = name
    					rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    					rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    				for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    					if np.isnan(rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')]):
    						rtn.loc[agentIndex[idFacility], ('Fuel', 'Commodity')] = name
    						rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')] = supply
    						rtn.loc[agentIndex[idFacility], ('Fuel', 'WasteFee')] = waste
    					else:
    						indice = rtn.index.size
    						rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    						rtn.loc[indice, ('Fuel', 'Commodity')] = name
    						rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    						rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
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
    				for idFacility in facilityIdList:
    					rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'FixedCost')] = fixed
    					rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'VariableCost')] = variable
    			fuel = prototype.find('fuel')
    			if fuel is not None:
    				for type in fuel.findall('type'):
    					supply = float(type.find('supply_cost').text)
    					waste = float(type.find('waste_fee').text)
    					name = type.find('name').text
    					for idFacility in facilityIdList:
    						if np.isnan(rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')]):
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'Commodity')] = name
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')] = supply
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'WasteFee')] = waste
    						else:
    							indice = rtn.index.size
    							rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    							rtn.loc[indice, ('Fuel', 'Commodity')] = name
    							rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    							rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
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
    				fuel = facility.find('fuel')
    				if fuel is not None:
    					for type in fuel.findall('type'):
    						supply = float(type.find('supply_cost').text)
    						waste = float(type.find('waste_fee').text)
    						name = type.find('name').text
    						if np.isnan(rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')]):
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'Commodity')] = name
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')] = supply
    							rtn.loc[agentIndex[idFacility], ('Fuel', 'WasteFee')] = waste
    						else:
    							indice = rtn.index.size
    							rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    							rtn.loc[indice, ('Fuel', 'Commodity')] = name
    							rtn.loc[indice, ('Fuel', 'SupplyCost')] = supply
    							rtn.loc[indice, ('Fuel', 'WasteFee')] = waste
    				decommissioning = facility.find('decommissioning')
    				if decommissioning is not None:
    					rtn.loc[agentIndex[idFacility], ('Decommissioning', 'Duration')] = int(decommissioning.find('duration').text)
    					rtn.loc[agentIndex[idFacility], ('Decommissioning', 'OvernightCost')] = float(decommissioning.find('overnight_cost').text)
    return rtn
	
del _eideps, _eischema