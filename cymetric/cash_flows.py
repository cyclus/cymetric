"""Functions to calculate more complex metrics that those calculated in eco_metrics.py. Some function are also dedicated to visualization of economic calculations (plotting of metrics). All metrics can be calculated at an agent, institution, region or simulation level.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cymetric.tools import dbopen
from cymetric.evaluator import Evaluator
from cymetric.eco_inputs import default_cap_overnight, default_discount_rate, default_fuel_price, actualization_vector, isreactor
import warnings
import os
        
####################################################################
# Calculation of average, annual and levalized cost of electricity #
####################################################################

# Reactor level

def annual_costs(outputDb, reactor_id, capital=True):
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
    commissioning = dfEntry[dfEntry.AgentId==reactor_id]['EnterTime'].iloc[0]
    dfCapitalCosts = evaler.eval('CapitalCost').reset_index()
    dfCapitalCosts = dfCapitalCosts[dfCapitalCosts.AgentId==reactor_id]
    dfCapitalCosts = dfCapitalCosts.groupby('Time').sum()
    costs = pd.DataFrame({'Capital' : dfCapitalCosts['Payment']}, index=list(range(0, duration)))
    dfDecommissioningCosts = evaler.eval('DecommissioningCost').reset_index()
    if not dfDecommissioningCosts.empty:
    	dfDecommissioningCosts = dfDecommissioningCosts[dfDecommissioningCosts.AgentId==reactor_id]
    	dfDecommissioningCosts = dfDecommissioningCosts.groupby('Time').sum()
    	costs['Decommissioning'] = dfDecommissioningCosts['Payment']
    dfOMCosts = evaler.eval('OperationMaintenance').reset_index()
    dfOMCosts = dfOMCosts[dfOMCosts.AgentId==reactor_id]
    dfOMCosts = dfOMCosts.groupby('Time').sum()
    costs['OandM'] = dfOMCosts['Payment']
    waste_disposal = 1
    dfFuelCosts = evaler.eval('FuelCost').reset_index()
    dfFuelCosts = dfFuelCosts[dfFuelCosts.AgentId==reactor_id]
    dfFuelCosts = dfFuelCosts.groupby('Time').sum()
    costs['Fuel'] = dfFuelCosts['Payment']
    costs = costs.fillna(0)
    costs['Year'] = (costs.index + initialMonth - 1) // 12 + initialYear
    if not capital:
    	del costs['Capital']
    costs = costs.groupby('Year').sum()
    return costs
    
def annual_costs_present_value(outputDb, reactor_id, capital=True):
	"""Same as annual_cost except all values are actualized to the begin date of the SIMULATION
	"""
	costs = annual_costs(outputDb, reactor_id, capital)
	actualization = actualization_vector(len(costs))
	actualization.index = costs.index
	return costs.apply(lambda x : x * actualization)
   
def average_cost(outputDb, reactor_id, capital=True):
    """Input : sqlite output database, reactor's AgentId
    Output : value (in $/MWh) corresponding to the total costs (sum of annual costs) divided by the total power generated.
    """
    db = dbopen(outputDb)
    evaler = Evaluator(db, write=False)
    f_power = evaler.eval('TimeSeriesPower').reset_index()
    power_generated = sum(f_power[f_power.AgentId==reactor_id]['Value']) * 8760 / 12
    return annual_costs(outputDb, reactor_id, capital).sum().sum() / power_generated
    
def cumulative_capital(outputDb, reactor_id):
	"""Input : sqlite output database and reactor agent id
	Output : cumulative sum of income and expense (= - expenditures + income)
	"""
	costs = - annual_costs(outputDb, reactor_id).sum(axis=1)
	power_gen = power_generated(outputDb, reactor_id) * lcoe(outputDb, reactor_id)
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
    
def lcoe(outputDb, reactor_id, capital=True):
	"""Input : sqlite output database and reactor agent id
	Output : Value corresponding to Levelized Cost of Electricity ($/MWh)
	"""
	costs = annual_costs(outputDb, reactor_id, capital)
	costs['TotalCosts'] = costs.sum(axis=1)
	commissioning = costs['Capital'].idxmax()
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	initialMonth = dfInfo['InitialMonth'].iloc[0]
	initialYear = dfInfo['InitialYear'].iloc[0]
	f_power = evaler.eval('TimeSeriesPower').reset_index()
	f_power = f_power[f_power.AgentId==reactor_id]
	f_power['Date'] = pd.Series(f_power.loc[:, 'Time']).apply(lambda x: (x + initialMonth - 1) // 12 + initialYear)
	del f_power['SimId']
	costs['Power'] = f_power.groupby('Date').sum()['Value'] * 8760 / 12
	costs = costs.fillna(0)
	power_generated = 0
	total_costs = 0
	for i in costs.index:
		power_generated += costs['Power'][i] / ((1 + default_discount_rate) ** (i - commissioning))
		total_costs += costs['TotalCosts'][i] / ((1 + default_discount_rate) ** (i - commissioning))
	return total_costs / power_generated 
	
def period_costs(outputDb, reactor_id, t0=0, period=20, capital=True):
	"""Input : sqlite output database, reactor id, time window (t0, period) 
	Output : cost at each time step t corresponding to actualized sum of expense in [t+t0, t+t0+period] divided by actualized power generated in [t+t0, t+t0+period]
	"""
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulationBegin') is not None:
				simulationBegin = int(truncation.find('simulationBegin').text)
			else:
				simulationBegin = 0
			if truncation.find('simulationEnd') is not None:
				simulationEnd = int(truncation.find('simulationEnd').text)
			else:
				simulationEnd = duration
	costs = annual_costs(outputDb, reactor_id, capital)
	costs = costs.sum(axis=1)
	power = power_generated(outputDb, reactor_id)
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
	
def period_costs2(outputDb, reactor_id, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
	"""
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	costs = annual_costs(outputDb, institution_id, capital)
	costs = costs.sum(axis=1)
	power = power_generated(outputDb, institution_id)
	df = pd.DataFrame(index=list(range(initialYear, initialYear + duration // 12 + 1)))
	df['Power'] = power
	df['Costs'] = costs
	df = df.fillna(0)
	simulationBegin = initialYear
	simulationEnd = (duration + initialMonth - 1) // 12 + initialYear
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
   
def power_generated(outputDb, reactor_id):
	"""Input : sqlite output database and reactor agent id
	Output : Electricity generated in MWh every years
	"""
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	f_power = evaler.eval('TimeSeriesPower').reset_index()	
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	f_power = f_power[f_power['AgentId']==reactor_id]
	f_power['Year'] = (f_power['Time'] + initialMonth - 1) // 12 + initialYear
	f_power = f_power.groupby('Year').sum()
	rtn = pd.Series(f_power['Value'] * 8760 / 12, index=list(range(initialYear, initialYear + (initialMonth + duration) // 12 + 1)))
	return rtn.fillna(0)
   
# Institution level
    
def institution_annual_costs(outputDb, institution_id, capital=True, truncate=True):
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
	dfEntry = dfEntry[dfEntry.ParentId==institution_id]
	dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
	dfPower = evaler.eval('TimeSeriesPower')
	id_reactor = dfEntry[dfEntry['AgentId'].apply(lambda x: isreactor(dfPower, x))]['AgentId'].tolist()
	dfCapitalCosts = evaler.eval('CapitalCost').reset_index()
	dfCapitalCosts = dfCapitalCosts[dfCapitalCosts['AgentId'].apply(lambda x: x in id_reactor)]
	dfCapitalCosts = dfCapitalCosts.groupby('Time').sum()
	costs = pd.DataFrame({'Capital' : dfCapitalCosts['Payment']}, index=list(range(0, duration)))
	dfDecommissioningCosts = evaler.eval('DecommissioningCost').reset_index()
	if not dfDecommissioningCosts.empty:
		dfDecommissioningCosts = dfDecommissioningCosts[dfDecommissioningCosts['AgentId'].apply(lambda x: x in id_reactor)]
		dfDecommissioningCosts = dfDecommissioningCosts.groupby('Time').sum()
		costs['Decommissioning'] = dfDecommissioningCosts['Payment']
	dfOMCosts = evaler.eval('OperationMaintenance').reset_index()
	dfOMCosts = dfOMCosts[dfOMCosts['AgentId'].apply(lambda x: x in id_reactor)]
	dfOMCosts = dfOMCosts.groupby('Time').sum()
	costs['OandM'] = dfOMCosts['Payment']
	waste_disposal = 1
	dfFuelCosts = evaler.eval('FuelCost').reset_index()
	dfFuelCosts = dfFuelCosts[dfFuelCosts['AgentId'].apply(lambda x: x in id_reactor)]
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
	
def institution_annual_costs_present_value(outputDb, reactor_id, capital=True):
	"""Same as annual_cost except all values are actualized to the begin date of the SIMULATION
	"""
	costs = institution_annual_costs(outputDb, institution_id, capital)
	actualization = actualization_vector(len(costs))
	actualization.index = costs.index
	return costs.apply(lambda x : x * actualization)
	
def institution_cumulative_capital(outputDb, institution_id):
	"""Input : sqlite output database and institution agent id
	Output : cumulative sum of income and expense (= - expenditures + income)
	"""
	costs = - institution_annual_costs(outputDb, institution_id).sum(axis=1)
	power_gen = institution_power_generated(outputDb, institution_id) * institution_average_lcoe(outputDb, institution_id)['Average LCOE']
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
		
def institution_period_costs(outputDb, institution_id, t0=0, period=20, capital=True):
	"""Input : sqlite output database, institution id, time window (t0, period) 
	Output : cost at each time step t corresponding to actualized sum of expense in [t+t0, t+t0+period] divided by actualized power generated in [t+t0, t+t0+period]
	"""
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulationBegin') is not None:
				simulationBegin = int(truncation.find('simulationBegin').text)
			else:
				simulationBegin = 0
			if truncation.find('simulationEnd') is not None:
				simulationEnd = int(truncation.find('simulationEnd').text)
			else:
				simulationEnd = duration
	costs = institution_annual_costs(outputDb, institution_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = institution_power_generated(outputDb, institution_id, truncate=False)
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
	
def institution_period_costs2(outputDb, institution_id, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
	"""
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulationBegin') is not None:
				simulationBegin = int(truncation.find('simulationBegin').text)
			else:
				simulationBegin = 0
			if truncation.find('simulationEnd') is not None:
				simulationEnd = int(truncation.find('simulationEnd').text)
			else:
				simulationEnd = duration
	costs = institution_annual_costs(outputDb, institution_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = institution_power_generated(outputDb, institution_id, truncate=False)
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
		
def institution_power_generated(outputDb, institution_id, truncate=True):
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
	dfEntry = dfEntry[dfEntry.ParentId==institution_id]
	dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
	id_reactor = dfEntry[dfEntry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	f_power = evaler.eval('TimeSeriesPower').reset_index()
	f_power = f_power[f_power['AgentId'].apply(lambda x: x in id_reactor)]
	f_power['Year'] = (f_power['Time'] + initialMonth - 1) // 12 + initialYear
	f_power = f_power.groupby('Year').sum()
	rtn = pd.Series(f_power['Value'] * 8760 / 12, index=list(range(initialYear, initialYear + (initialMonth + duration) // 12 + 1)))
	rtn.name = 'Power in MWh'
	return rtn.fillna(0)

def institution_lcoe(outputDb, institution_id):
	"""Input : sqlite output database and institution agent id
	Output : Value corresponding to Levelized Cost of Electricity ($/MWh)
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
	costs = institution_annual_costs(outputDb, institution_id, truncate=False)
	costs['TotalCosts'] = costs.sum(axis=1)
	commissioning = costs['Capital'].idxmax()
	costs['Power'] = institution_power_generated(outputDb, institution_id)
	costs = costs.fillna(0)
	power_generated = 0
	total_costs = 0
	for i in costs.index:
		power_generated += costs['Power'][i] / ((1 + default_discount_rate) ** (i - commissioning))
		total_costs += costs['TotalCosts'][i] / ((1 + default_discount_rate) ** (i - commissioning))
	return total_costs / power_generated

def institution_average_lcoe(outputDb, institution_id):
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
	dfEntry = dfEntry[dfEntry.ParentId==institution_id]
	dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
	dfPower = evaler.eval('TimeSeriesPower')
	id_reactor = dfEntry[dfEntry['Spec'].apply(lambda x: isreactor(dfPower, x))]['AgentId'].tolist()
	simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear # year instead of months
	simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
	f_power = evaler.eval('TimeSeriesPower')
	rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
	rtn['Weighted sum'] = 0
	rtn['Power'] = 0
	rtn['Temp'] = pd.Series()
	rtn['Temp2'] = pd.Series()
	for id in id_reactor:
		tmp = lcoe(outputDb, id)
		commissioning = dfEntry[dfEntry.AgentId==id]['EnterTime'].iloc[0]
		lifetime = dfEntry[dfEntry.AgentId==id]['Lifetime'].iloc[0]
		decommissioning = (commissioning + lifetime + initialMonth - 1) // 12 + initialYear
		commissioning = (commissioning + initialMonth - 1) // 12 + initialYear
		power = f_power[f_power.AgentId==id]['Value'].iloc[0]
		rtn['Temp'] = pd.Series(tmp, index=list(range(commissioning, decommissioning + 1))) * power
		rtn['Weighted sum'] += rtn['Temp'].fillna(0)
		rtn['Temp2'] = pd.Series(power, index=list(range(commissioning, decommissioning + 1))).fillna(0)
		rtn['Power'] += rtn['Temp2'].fillna(0)
	rtn['Average LCOE'] = rtn['Weighted sum'] / rtn['Power']
	return rtn.fillna(0)
		
# Region level

def region_annual_costs(outputDb, region_id, capital=True, truncate=True):
	"""Input : sqlite output database and region's AgentId. It is possible not to take into account the construction costs (capital=False) if the reactors are supposed to have been built before the beginning of the simulation. It is also possible to truncate the simulation results and only have access to cash flows occurring between the two dates (begin and end) specified in 'parameters.xml'. The truncation allows to let reactors decommission after the end of the simulation and thus to take into account cash flows that occur after the end of the simulation for example to calculate the LCOE.
	Output : total reactor costs per year over its lifetime at the region level.
	"""
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulationBegin') is not None:
				simulationBegin = int(truncation.find('simulationBegin').text)
			else:
				simulationBegin = 0
			if truncation.find('simulationEnd') is not None:
				simulationEnd = int(truncation.find('simulationEnd').text)
			else:
				simulationEnd = duration
	dfEntry = evaler.eval('AgentEntry').reset_index()
	tmp = dfEntry[dfEntry.ParentId==region_id]
	dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
	id_inst = tmp[tmp.Kind=='Inst']['AgentId'].tolist()
	id_reactor = []
	for id in id_inst:
		dfEntry2 = dfEntry[dfEntry.ParentId==id]
		id_reactor += dfEntry2[dfEntry2['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	dfCapitalCosts = evaler.eval('CapitalCost').reset_index()
	dfCapitalCosts = dfCapitalCosts[dfCapitalCosts['AgentId'].apply(lambda x: x in id_reactor)]
	mini = min(dfCapitalCosts['Time'])
	dfCapitalCosts = dfCapitalCosts.groupby('Time').sum()
	costs = pd.DataFrame({'Capital' : dfCapitalCosts['Payment']}, index=list(range(mini, duration)))
	dfDecommissioningCosts = evaler.eval('DecommissioningCost').reset_index()
	if not dfDecommissioningCosts.empty:
		dfDecommissioningCosts = dfDecommissioningCosts[dfDecommissioningCosts['AgentId'].apply(lambda x: x in id_reactor)]
		dfDecommissioningCosts = dfDecommissioningCosts.groupby('Time').sum()
		costs['Decommissioning'] = dfDecommissioningCosts['Payment']
	dfOMCosts = evaler.eval('OperationMaintenance').reset_index()
	dfOMCosts = dfOMCosts[dfOMCosts['AgentId'].apply(lambda x: x in id_reactor)]
	dfOMCosts = dfOMCosts.groupby('Time').sum()
	costs['OandM'] = dfOMCosts['Payment']
	waste_disposal = 1
	dfFuelCosts = evaler.eval('FuelCost').reset_index()
	dfFuelCosts = dfFuelCosts[dfFuelCosts['AgentId'].apply(lambda x: x in id_reactor)]
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
		
def region_annual_costs_present_value(outputDb, region_id, capital=True, truncate=True):
	"""Same as annual_cost except all values are actualized to the begin date of the SIMULATION
	"""
	costs = region_annual_costs(outputDb, region_id, capital)
	actualization = actualization_vector(len(costs))
	actualization.index = costs.index
	return costs.apply(lambda x : x * actualization)
		
def region_cumulative_capital(outputDb, region_id):
	"""Input : sqlite output database and region agent id
	Output : cumulative sum of income and expense (= - expenditures + income)
	"""
	costs = - region_annual_costs(outputDb, region_id).sum(axis=1)
	power_gen = region_power_generated(outputDb, region_id) * region_average_lcoe(outputDb, region_id)['Average LCOE']
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
		
def region_period_costs(outputDb, region_id, t0=0, period=20, capital=True):
	"""Input : sqlite output database, region id, time window (t0, period) 
	Output : cost at each time step t corresponding to actualized sum of expense in [t+t0, t+t0+period] divided by actualized power generated in [t+t0, t+t0+period]
	"""
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulationBegin') is not None:
				simulationBegin = int(truncation.find('simulationBegin').text)
			else:
				simulationBegin = 0
			if truncation.find('simulationEnd') is not None:
				simulationEnd = int(truncation.find('simulationEnd').text)
			else:
				simulationEnd = duration
	costs = region_annual_costs(outputDb, region_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = region_power_generated(outputDb, region_id, truncate=False)
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
	
def region_period_costs2(outputDb, region_id, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
	"""
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulationBegin') is not None:
				simulationBegin = int(truncation.find('simulationBegin').text)
			else:
				simulationBegin = 0
			if truncation.find('simulationEnd') is not None:
				simulationEnd = int(truncation.find('simulationEnd').text)
			else:
				simulationEnd = duration
	costs = region_annual_costs(outputDb, region_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = region_power_generated(outputDb, region_id, truncate=False)
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
		
def region_power_generated(outputDb, region_id, truncate=True):
	"""Input : sqlite output database and region agent id
	Output : Sum of electricity generated in MWh every years in the region reactor fleet
	"""
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulationBegin') is not None:
				simulationBegin = int(truncation.find('simulationBegin').text)
			else:
				simulationBegin = 0
			if truncation.find('simulationEnd') is not None:
				simulationEnd = int(truncation.find('simulationEnd').text)
			else:
				simulationEnd = duration
	dfEntry = evaler.eval('AgentEntry').reset_index()
	tmp = dfEntry[dfEntry.ParentId==region_id]
	dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
	id_inst = tmp[tmp.Kind=='Inst']['AgentId'].tolist()
	id_reactor = []
	for id in id_inst:
		dfEntry2 = dfEntry[dfEntry.ParentId==id]
		id_reactor += dfEntry2[dfEntry2['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	f_power = evaler.eval('TimeSeriesPower').reset_index()
	f_power = f_power[f_power['AgentId'].apply(lambda x: x in id_reactor)]
	f_power['Year'] = (f_power['Time'] + initialMonth - 1) // 12 + initialYear
	f_power = f_power.groupby('Year').sum()
	rtn = pd.Series(f_power['Value'] * 8760 / 12, index=list(range(initialYear, initialYear + (initialMonth + duration) // 12 + 1)))
	rtn.name = 'Power in MWh'
	return rtn.fillna(0)

def region_lcoe(outputDb, region_id):
	"""Input : sqlite output database and region agent id
	Output : Value corresponding to Levelized Cost of Electricity ($/MWh)
	"""
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulationBegin') is not None:
				simulationBegin = int(truncation.find('simulationBegin').text)
			else:
				simulationBegin = 0
			if truncation.find('simulationEnd') is not None:
				simulationEnd = int(truncation.find('simulationEnd').text)
			else:
				simulationEnd = duration
	costs = region_annual_costs(outputDb, region_id, truncate=False)
	costs['TotalCosts'] = costs.sum(axis=1)
	commissioning = costs['Capital'].idxmax()
	costs['Power'] = region_power_generated(outputDb, region_id)
	costs = costs.fillna(0)
	power_generated = 0
	total_costs = 0
	for i in costs.index:
		power_generated += costs['Power'][i] / ((1 + default_discount_rate) ** (i - commissioning))
		total_costs += costs['TotalCosts'][i] / ((1 + default_discount_rate) ** (i - commissioning))
	return total_costs / power_generated

def region_average_lcoe(outputDb, region_id):
	"""Input : sqlite output database and region agent id
	Output : Variable cost corresponding at each time step (i.e. every year) to the weighted average of the reactors Levelized Cost of Electricity ($/MWh). A reactor is taken into account at time step t if and only if it is in activity (i.e. already commissioned and not yet decommissioned) at time step t.
	"""
	db = dbopen(outputDb)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulationBegin') is not None:
				simulationBegin = int(truncation.find('simulationBegin').text)
			else:
				simulationBegin = 0
			if truncation.find('simulationEnd') is not None:
				simulationEnd = int(truncation.find('simulationEnd').text)
			else:
				simulationEnd = duration
	dfEntry = evaler.eval('AgentEntry').reset_index()
	tmp = dfEntry[dfEntry.ParentId==region_id]
	dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
	id_inst = tmp[tmp.Kind=='Inst']['AgentId'].tolist()
	id_reactor = []
	f_power = evaler.eval('TimeSeriesPower')
	for id in id_inst:
		dfEntry2 = dfEntry[dfEntry.ParentId==id]
		id_reactor += dfEntry2[dfEntry2['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear # year instead of months
	simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
	rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
	rtn['Weighted sum'] = 0
	rtn['Power'] = 0
	rtn['Temp'] = pd.Series()
	rtn['Temp2'] = pd.Series()
	for id in id_reactor:
		tmp = lcoe(outputDb, id)
		commissioning = dfEntry[dfEntry.AgentId==id]['EnterTime'].iloc[0]
		lifetime = dfEntry[dfEntry.AgentId==id]['Lifetime'].iloc[0]
		decommissioning = (commissioning + lifetime + initialMonth - 1) // 12 + initialYear
		commissioning = (commissioning + initialMonth - 1) // 12 + initialYear
		power = f_power[f_power.AgentId==id]['Value'].iloc[0]
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
	id_reactor = dfEntry[dfEntry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	dfCapitalCosts = evaler.eval('CapitalCost').reset_index()
	dfCapitalCosts = dfCapitalCosts[dfCapitalCosts['AgentId'].apply(lambda x: x in id_reactor)]
	mini = min(dfCapitalCosts['Time'])
	dfCapitalCosts = dfCapitalCosts.groupby('Time').sum()
	costs = pd.DataFrame({'Capital' : dfCapitalCosts['Payment']}, index=list(range(0, duration)))
	dfDecommissioningCosts = evaler.eval('DecommissioningCost').reset_index()
	if not dfDecommissioningCosts.empty:
		dfDecommissioningCosts = dfDecommissioningCosts[dfDecommissioningCosts['AgentId'].apply(lambda x: x in id_reactor)]
		dfDecommissioningCosts = dfDecommissioningCosts.groupby('Time').sum()
		costs['Decommissioning'] = dfDecommissioningCosts['Payment']
	dfOMCosts = evaler.eval('OperationMaintenance').reset_index()
	dfOMCosts = dfOMCosts[dfOMCosts['AgentId'].apply(lambda x: x in id_reactor)]
	dfOMCosts = dfOMCosts.groupby('Time').sum()
	costs['OandM'] = dfOMCosts['Payment']
	waste_disposal = 1
	dfFuelCosts = evaler.eval('FuelCost').reset_index()
	dfFuelCosts = dfFuelCosts[dfFuelCosts['AgentId'].apply(lambda x: x in id_reactor)]
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
	
def simulation_cumulative_capital(outputDb, annualCosts=-1, powerGenerated=-1):
	"""Input : sqlite output database
	Output : cumulative sum of total income and total expense (= - expenditures + income) when all reactors of the simulation are taken into account
	"""
	if annualCosts is -1 :
		costs = - simulation_annual_costs(outputDb).sum(axis=1)
	else:
		costs = annualCosts
	if powerGenerated is -1 :
		power_gen = simulation_power_generated(outputDb) * simulation_average_lcoe(outputDb)['Average LCOE']
	else:
		power_gen = powerGenerated
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
		
def simulation_period_costs(outputDb, t0=0, period=20, capital=True, annualCosts=-1, powerGenerated=-1):
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
	if annualCosts is -1 :
		costs = simulation_annual_costs(outputDb, capital, truncate=False).sum(axis=1)
	else:
		costs = annualCosts
	if powerGenerated is -1 :
		power = simulation_power_generated(outputDb, truncate=False)
	else:
		power = powerGenerated
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
	
def simulation_period_costs2(outputDb, t0=0, period=20, capital=True, annualCosts=-1, powerGenerated=-1):
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
	if annualCosts is -1 :
		costs = simulation_annual_costs(outputDb, capital, truncate=False).sum(axis=1)
	else:
		costs = annualCosts
	if powerGenerated is -1 :
		power = simulation_power_generated(outputDb, truncate=False)
	else:
		power = powerGenerated
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
	id_reactor = dfEntry[dfEntry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	f_power = evaler.eval('TimeSeriesPower').reset_index()
	f_power = f_power[f_power['AgentId'].apply(lambda x: x in id_reactor)]
	f_power['Year'] = (f_power['Time'] + initialMonth - 1) // 12 + initialYear
	f_power = f_power.groupby('Year').sum()
	rtn = pd.Series(f_power['Value'] * 8760 / 12, index=list(range(initialYear, initialYear + (initialMonth + duration) // 12 + 1)))
	rtn.name = 'Power in MWh'
	return rtn.fillna(0)

def simulation_lcoe(outputDb, annualCosts=-1, powerGenerated=-1):
	"""Input : sqlite output database
	Output : Value corresponding to Levelized Cost of Electricity ($/MWh) when taking into account all reactors commissioned in the simulation
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
	if annualCosts is -1:
		costs = simulation_annual_costs(outputDb, truncate=False).sum(axis=1)
	else:
		costs = annualCosts
	commissioning = costs.idxmax()
	if powerGenerated is -1:
		power = simulation_power_generated(outputDb)
	else:
		power = powerGenerated
	costs = costs.fillna(0)
	power_generated = 0
	total_costs = 0
	for i in costs.index:
		power_generated += power[i] / ((1 + default_discount_rate) ** (i - commissioning))
		total_costs += costs[i] / ((1 + default_discount_rate) ** (i - commissioning))
	return total_costs / power_generated

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
	id_reactor = dfEntry[dfEntry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	simulationBegin = (simulationBegin + initialMonth - 1) // 12 + initialYear # year instead of months
	simulationEnd = (simulationEnd + initialMonth - 1) // 12 + initialYear
	f_power = evaler.eval('TimeSeriesPower')
	rtn = pd.DataFrame(index=list(range(simulationBegin, simulationEnd + 1)))
	rtn['Weighted sum'] = 0
	rtn['Power'] = 0
	rtn['Temp'] = pd.Series()
	rtn['Temp2'] = pd.Series()
	for id in id_reactor:
		tmp = lcoe(outputDb, id)
		commissioning = dfEntry[dfEntry.AgentId==id]['EnterTime'].iloc[0]
		lifetime = dfEntry[dfEntry.AgentId==id]['Lifetime'].iloc[0]
		decommissioning = (commissioning + lifetime + initialMonth - 1) // 12 + initialYear
		commissioning = (commissioning + initialMonth - 1) // 12 + initialYear
		power = f_power[f_power.AgentId==id]['Value'].iloc[0]
		rtn['Temp'] = pd.Series(tmp, index=list(range(commissioning, decommissioning + 1))) * power
		rtn['Weighted sum'] += rtn['Temp'].fillna(0)
		rtn['Temp2'] = pd.Series(power, index=list(range(commissioning, decommissioning + 1)))
		rtn['Power'] += rtn['Temp2'].fillna(0)
		print(id) # test
	rtn['Average LCOE'] = rtn['Weighted sum'] / rtn['Power']
	return rtn.fillna(0)	