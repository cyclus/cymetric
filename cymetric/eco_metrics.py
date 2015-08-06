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


_ccdeps = [('TimeSeriesPower', ('SimId', 'AgentId', 'Value'), 'Time'), ('AgentEntry', ('AgentId', 'ParentId', 'Spec'), 'EnterTime'), ('Info', ('InitialYear', 'InitialMonth'), 'Duration'), ('EconomicInfo', (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Capital', 'Begin'), ('Capital', 'Duration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost')), ('Finance','DiscountRate'))]

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
    tuples = (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Capital', 'Begin'), ('Capital', 'Duration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost'), ('Finance','DiscountRate'))
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
    	tmp = dfEcoInfo.loc[id]
    	if 'REACTOR' in tmp.loc[('Agent', 'Prototype')].upper():
    		deviation = tmp.loc[('Capital', 'Deviation')]
    		variance = deviation ** 2
    		deviation = np.random.poisson(variance) - variance
    		begin = int(tmp.loc[('Capital', 'Begin')] + deviation)
    		duration = int(tmp.loc[('Capital', 'Duration')] + 2 * deviation)
    		overnightCost = tmp.loc[('Capital', 'OvernightCost')]
    		cashFlowShape = capital_shape(begin, duration)
    		powerCapacity = max(dfPower[dfPower.AgentId==id]['Value'])
    		discountRate = tmp.loc[('Finance','DiscountRate')]
    		cashFlow = np.around(cashFlowShape * overnightCost * powerCapacity, 3)
    		cashFlow *= ((1 + discountRate) ** math.ceil(duration / 12) - 1) / (discountRate * math.ceil(duration / 12))
    		tmp = pd.DataFrame({'AgentId': id, 'Time': pd.Series(list(range(duration + 1))) - begin + dfEntry.EnterTime[id], 'Payment' : cashFlow})
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
    	tmpEcoInfo = dfEcoInfo.loc[agentId]
    	tmpTrans = dfTransactions[dfTransactions.ReceiverId==agentId]
    	for commod in tmpEcoInfo.loc[('Fuel', 'SupplyCost')]:
    		price = tmpEcoInfo.loc[('Fuel', 'SupplyCost')][commod]
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
    	duration = int(dfEcoInfo.loc[id, ('Decommissioning', 'Duration')])
    	overnightCost = dfEcoInfo.loc[id, ('Decommissioning', 'OvernightCost')]
    	cashFlowShape = capital_shape(duration // 2, duration-1)
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
    	if isreactor(id, rtn):
    		powerGenerated = rtn[rtn.AgentId==id].loc[:,'Value']
    		powerCapacity = max(powerGenerated)
    		powerGenerated *= 8760 / 12
    		fixedOM = dfEcoInfo.loc[id, ('OperationMaintenance', 'FixedCost')]
    		variableOM = dfEcoInfo.loc[id, ('OperationMaintenance', 'VariableCost')]
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
    tuples = [('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Agent', 'ParentId'), ('Finance','ReturnOnDebt'), ('Finance','ReturnOnEquity'), ('Finance','TaxRate'), ('Finance','DiscountRate'), ('Capital', 'Begin'), ('Capital', 'Duration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost'), ('Decommissioning', 'Duration'), ('Decommissioning', 'OvernightCost'), ('OperationMaintenance', 'FixedCost'), ('OperationMaintenance', 'VariableCost'), ('Fuel', 'SupplyCost'), ('Fuel', 'WasteFee'), ('Truncation', 'Begin'), ('Truncation', 'End')]
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
    	rtn.loc[:, ('Finance','ReturnOnDebt')] = float(finance.find('return_on_debt').text)
    	rtn.loc[:, ('Finance','ReturnOnEquity')] = float(finance.find('return_on_equity').text)
    	rtn.loc[:, ('Finance','DiscountRate')] = float(finance.find('discount_rate').text)
    capital = root.find('capital')
    if not capital == None:
    	rtn.loc[:, ('Capital', 'Begin')] = int(capital.find('begin').text)
    	rtn.loc[:, ('Capital', 'Duration')] = int(capital.find('duration').text)
    	rtn.loc[:, ('Capital', 'Deviation')] = float(capital.find('deviation').text)
    	rtn.loc[:, ('Capital', 'OvernightCost')] = int(capital.find('overnight_cost').text)
    decommissioning = root.find('decommissioning')
    if not decommissioning == None:
    	rtn.loc[:, ('Decommissioning', 'Duration')] = int(decommissioning.find('duration').text)
    	rtn.loc[:, ('Decommissioning', 'OvernightCost')] = int(decommissioning.find('overnight_cost').text)
    operation_maintenance = root.find('operation_maintenance')
    if not operation_maintenance == None:
    	rtn.loc[:, ('OperationMaintenance', 'FixedCost')] = int(operation_maintenance.find('fixed').text)
    	rtn.loc[:, ('OperationMaintenance', 'VariableCost')] = int(operation_maintenance.find('variable').text)
    fuel = root.find('fuel')
    dfSupply = pd.DataFrame(index=rtn.index)
    dfSupply['SupplyCost'] = pd.Series()
    dfWaste = pd.DataFrame(index=rtn.index)
    dfWaste['WasteFee'] = pd.Series()
    supply = {}
    waste = {}
    if not fuel == None:
    	for type in fuel.findall('type'):
    		supply[type.find('name').text] = int(type.find('supply_cost').text)
    		waste[type.find('name').text] = int(type.find('waste_fee').text)
    	supply = tools.hashabledict(supply)
    	waste = tools.hashabledict(waste)
    	for j in rtn.index:
    		dfSupply.loc[j, 'SupplyCost'] = supply
    		dfWaste.loc[j, 'WasteFee'] = waste
    	rtn.loc[:, ('Fuel', 'SupplyCost')] = dfSupply.loc[:, 'SupplyCost']
    	rtn.loc[:, ('Fuel', 'WasteFee')] = dfWaste.loc[:, 'WasteFee']
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
    		begin = int(capital.find('begin').text)
    		duration = int(capital.find('duration').text)
    		deviation = float(capital.find('deviation').text)
    		overnightCost = int(capital.find('overnight_cost').text)
    		rtn.loc[agentIndex[idRegion], ('Capital', 'Begin')] = begin
    		rtn.loc[agentIndex[idRegion], ('Capital', 'Duration')] = duration
    		rtn.loc[agentIndex[idRegion], ('Capital', 'Deviation')] = deviation
    		rtn.loc[agentIndex[idRegion], ('Capital', 'OvernightCost')] = overnightCost
    		for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Begin')] = begin
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Duration')] = duration
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Deviation')] = deviation
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'OvernightCost')] = overnightCost
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('Capital', 'Begin')] = begin
    				rtn.loc[agentIndex[idFacility], ('Capital', 'Duration')] = duration
    				rtn.loc[agentIndex[idFacility], ('Capital', 'Deviation')] = deviation
    				rtn.loc[agentIndex[idFacility], ('Capital', 'OvernightCost')] = overnightCost
    	decommissioning = region.find('decommissioning')
    	if decommissioning is not None:
    		duration = int(decommissioning.find('duration').text)
    		overnightCost = int(decommissioning.find('overnight_cost').text)
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
    		fixed = int(operation_maintenance.find('fixed').text)
    		variable = int(operation_maintenance.find('variable').text)
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
    		supply = {}
    		waste = {}
    		for type in fuel.findall('type'):
    			supply[type.find('name').text] = int(type.find('supply_cost').text)
    			waste[type.find('name').text] = int(type.find('waste_fee').text)
    		supply = tools.hashabledict(supply)
    		waste = tools.hashabledict(waste)
    		dfSupply.loc[agentIndex[idRegion], 'SupplyCost'] = supply
    		dfWaste.loc[agentIndex[idRegion], 'WasteFee'] = waste
    		rtn.loc[agentIndex[idRegion], ('Fuel', 'SupplyCost')] = dfSupply.loc[agentIndex[idRegion], 'SupplyCost']
    		rtn.loc[agentIndex[idRegion], ('Fuel', 'WasteFee')] = dfWaste.loc[agentIndex[idRegion], 'WasteFee']
    		for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    			dfSupply.loc[agentIndex[idInstitution], 'SupplyCost'] = supply
    			dfWaste.loc[agentIndex[idInstitution], 'WasteFee'] = waste
    			rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')] = dfSupply.loc[agentIndex[idInstitution], 'SupplyCost']
    			rtn.loc[agentIndex[idInstitution], ('Fuel', 'WasteFee')] = dfWaste.loc[agentIndex[idInstitution], 'WasteFee']	
    		for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    			dfSupply.loc[agentIndex[idFacility], 'SupplyCost'] = supply
    			dfWaste.loc[agentIndex[idFacility], 'WasteFee'] = waste
    			rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')] = dfSupply.loc[agentIndex[idFacility], 'SupplyCost']
    			rtn.loc[agentIndex[idFacility], ('Fuel', 'WasteFee')] = dfWaste.loc[agentIndex[idFacility], 'WasteFee']
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
    			begin = int(capital.find('begin').text)
    			duration = int(capital.find('duration').text)
    			deviation = float(capital.find('deviation').text)
    			overnightCost = int(capital.find('overnight_cost').text)
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Begin')] = begin
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Duration')] = duration
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Deviation')] = deviation
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'OvernightCost')] = overnightCost
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('Capital', 'Begin')] = begin
    				rtn.loc[agentIndex[idFacility], ('Capital', 'Duration')] = duration
    				rtn.loc[agentIndex[idFacility], ('Capital', 'Deviation')] = deviation
    				rtn.loc[agentIndex[idFacility], ('Capital', 'OvernightCost')] = overnightCost
    		decommissioning = institution.find('decommissioning')
    		if decommissioning is not None:
    			duration = int(decommissioning.find('duration').text)
    			overnightCost = int(decommissioning.find('overnight_cost').text)
    			rtn.loc[agentIndex[idInstitution], ('Decommissioning', 'Duration')] = duration
    			rtn.loc[agentIndex[idInstitution], ('Decommissioning', 'OvernightCost')] = overnightCost
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('Decommissioning', 'Duration')] = duration
    				rtn.loc[agentIndex[idFacility], ('Decommissioning', 'OvernightCost')] = overnightCost
    		operation_maintenance = institution.find('operation_maintenance')
    		if operation_maintenance is not None:
    			fixed = int(operation_maintenance.find('fixed').text)
    			variable = int(operation_maintenance.find('variable').text)
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'FixedCost')] = fixed
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'VariableCost')] = variable
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'FixedCost')] = fixed
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'VariableCost')] = variable
    		fuel = institution.find('fuel')
    		if fuel is not None:
    			supply = {}
    			waste = {}
    			for type in supply.findall('type'):
    				supply[type.find('name').text] = int(type.find('supply_cost').text)
    				waste[type.find('name').text] = int(type.find('waste_fee').text)
    			supply = tools.hashabledict(supply)
    			waste = tools.hashabledict(waste)
    			dfSupply.loc[agentIndex[idInstitution], 'SupplyCost'] = supply
    			dfWaste.loc[agentIndex[idInstitution], 'WasteFee'] = waste
    			rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')] = dfSupply.loc[agentIndex[idInstitution], 'SupplyCost']
    			rtn.loc[agentIndex[idInstitution], ('Fuel', 'WasteFee')] = dfWaste.loc[agentIndex[idInstitution], 'WasteFee']
    			for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    				dfSupply.loc[agentIndex[idFacility], 'SupplyCost'] = supply
    				dfWaste.loc[agentIndex[idFacility], 'WasteFee'] = waste
    				rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')] = dfSupply.loc[agentIndex[idFacility], 'SupplyCost']
    				rtn.loc[agentIndex[idFacility], ('Fuel', 'WasteFee')] = dfWaste.loc[agentIndex[idFacility], 'WasteFee']
    		for facility in institution.findall('facility'):
    			idFacility = int(facility.find('id').text)
    			capital = facility.find('capital')
    			if capital is not None:
    				rtn.loc[agentIndex[idFacility], ('Capital', 'Begin')] = int(capital.find('begin').text)
    				rtn.loc[agentIndex[idFacility], ('Capital', 'Duration')] = int(capital.find('duration').text)
    				rtn.loc[agentIndex[idFacility], ('Capital', 'Deviation')] = float(capital.find('deviation').text)
    				rtn.loc[agentIndex[idFacility], ('Capital', 'OvernightCost')] = int(capital.find('overnight_cost').text)
    			operation_maintenance = facility.find('operation_maintenance')
    			if operation_maintenance is not None:
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'FixedCost')] = int(operation_maintenance.find('fixed').text)
    				rtn.loc[agentIndex[idFacility], ('OperationMaintenance', 'VariableCost')] = int(operation_maintenance.find('variable').text)
    			fuel = facility.find('fuel')
    			if fuel is not None:
    				supply = {}
    				waste = {}
    				for type in supply.findall('type'):
    					supply[type.find('name').text] = int(type.find('supply_cost').text)
    					waste[type.find('name').text] = int(type.find('waste_fee').text)
    				supply = tools.hashabledict(supply)
    				waste = tools.hashabledict(waste)
    				dfSupply.loc[agentIndex[idFacility], 'SupplyCost'] = supply
    				dfWaste.loc[agentIndex[idFacility], 'WasteFee'] = waste
    				rtn.loc[agentIndex[idFacility], ('Fuel', 'SupplyCost')] = dfSupply.loc[agentIndex[idReactor], 'SupplyCost']
    				rtn.loc[agentIndex[idFacility], ('Fuel', 'WasteFee')] = dfWaste.loc[agentIndex[idReactor], 'WasteFee']
    			decommissioning = facility.find('decommissioning')
    			if decommissioning is not None:
    				rtn.loc[agentIndex[idFacility], ('Decommissioning', 'Duration')] = int(decommissioning.find('duration').text)
    				rtn.loc[agentIndex[idFacility], ('Decommissioning', 'OvernightCost')] = int(decommissioning.find('overnight_cost').text)
    return rtn
	
del _eideps, _eischema