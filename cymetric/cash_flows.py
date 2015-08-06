"""Functions to calculate more complex metrics that those calculated in eco_metrics.py. Some function are also dedicated to visualization of economic calculations (plotting of metrics). All metrics can be calculated at an agent, institution, region or simulation level.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cymetric.tools import dbopen
from cymetric.evaluator import Evaluator
from cymetric.eco_inputs import default_cap_overnight, default_discount_rate, default_fuel_price, actualization_vector
import warnings
import os
        
####################################################################
# Calculation of average, annual and levalized cost of electricity #
####################################################################

# Reactor level

def annual_costs(output_db, reactor_id, capital=True):
    """Input : sqlite output database and reactor's AgentId. It is possible to not take into account the construction costs (capital=False) if the reactor is supposed to have been built before the beginning of the simulation.
    Output : total reactor costs per year over its lifetime.
    """
    db = dbopen(output_db)
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
    
def annual_costs_present_value(output_db, reactor_id, capital=True):
	"""Same as annual_cost except all values are actualized to the begin date of the SIMULATION
	"""
	costs = annual_costs(output_db, reactor_id, capital)
	actualization = actualization_vector(len(costs))
	actualization.index = costs.index
	return costs.apply(lambda x : x * actualization)
   
def average_cost(output_db, reactor_id, capital=True):
    """Input : sqlite output database, reactor's AgentId
    Output : value (in $/MWh) corresponding to the total costs (sum of annual costs) divided by the total power generated.
    """
    db = dbopen(output_db)
    evaler = Evaluator(db, write=False)
    f_power = evaler.eval('TimeSeriesPower').reset_index()
    power_generated = sum(f_power[f_power.AgentId==reactor_id]['Value']) * 8760 / 12
    return annual_costs(output_db, reactor_id, capital).sum().sum() / power_generated
    
def cumulative_capital(output_db, reactor_id):
	"""Input : sqlite output database and reactor agent id
	Output : cumulative sum of income and expense (= - expenditures + income)
	"""
	costs = - annual_costs(output_db, reactor_id).sum(axis=1)
	power_gen = power_generated(output_db, reactor_id) * lcoe(output_db, reactor_id)
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
    
def lcoe(output_db, reactor_id, capital=True):
	"""Input : sqlite output database and reactor agent id
	Output : Value corresponding to Levelized Cost of Electricity ($/MWh)
	"""
	costs = annual_costs(output_db, reactor_id, capital)
	costs['TotalCosts'] = costs.sum(axis=1)
	commissioning = costs['Capital'].idxmax()
	db = dbopen(output_db)
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
	
def period_costs(output_db, reactor_id, t0=0, period=20, capital=True):
	"""Input : sqlite output database, reactor id, time window (t0, period) 
	Output : cost at each time step t corresponding to actualized sum of expense in [t+t0, t+t0+period] divided by actualized power generated in [t+t0, t+t0+period]
	"""
	db = dbopen(output_db)
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
	costs = annual_costs(output_db, reactor_id, capital)
	costs = costs.sum(axis=1)
	power = power_generated(output_db, reactor_id)
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
	
def period_costs2(output_db, reactor_id, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	costs = annual_costs(output_db, institution_id, capital)
	costs = costs.sum(axis=1)
	power = power_generated(output_db, institution_id)
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
   
def power_generated(output_db, reactor_id):
	"""Input : sqlite output database and reactor agent id
	Output : Electricity generated in MWh every years
	"""
	db = dbopen(output_db)
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
    
def institution_annual_costs(output_db, institution_id, capital=True, truncate=True):
	"""Input : sqlite output database and institution's AgentId. It is possible not to take into account the construction costs (capital=False) if the reactors are supposed to have been built before the beginning of the simulation. It is also possible to truncate the simulation results and only have access to cash flows occurring between the two dates (begin and end) specified in 'parameters.xml'. The truncation allows to let reactors decommission after the end of the simulation and thus to take into account cash flows that occur after the end of the simulation for example to calculate the LCOE.
	Output : total reactor costs per year over its lifetime at the institution level.
	"""
	db = dbopen(output_db)
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
	dfEntry = dfEntry[dfEntry.ParentId==institution_id]
	dfEntry = dfEntry[dfEntry['EnterTime'].apply(lambda x: x>simulationBegin and x<simulationEnd)]
	id_reactor = dfEntry[dfEntry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
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
	
def institution_annual_costs_present_value(output_db, reactor_id, capital=True):
	"""Same as annual_cost except all values are actualized to the begin date of the SIMULATION
	"""
	costs = institution_annual_costs(output_db, institution_id, capital)
	actualization = actualization_vector(len(costs))
	actualization.index = costs.index
	return costs.apply(lambda x : x * actualization)
	
def institution_cumulative_capital(output_db, institution_id):
	"""Input : sqlite output database and institution agent id
	Output : cumulative sum of income and expense (= - expenditures + income)
	"""
	costs = - institution_annual_costs(output_db, institution_id).sum(axis=1)
	power_gen = institution_power_generated(output_db, institution_id) * institution_average_lcoe(output_db, institution_id)['Average LCOE']
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
		
def institution_period_costs(output_db, institution_id, t0=0, period=20, capital=True):
	"""Input : sqlite output database, institution id, time window (t0, period) 
	Output : cost at each time step t corresponding to actualized sum of expense in [t+t0, t+t0+period] divided by actualized power generated in [t+t0, t+t0+period]
	"""
	db = dbopen(output_db)
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
	costs = institution_annual_costs(output_db, institution_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = institution_power_generated(output_db, institution_id, truncate=False)
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
	
def institution_period_costs2(output_db, institution_id, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
	"""
	db = dbopen(output_db)
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
	costs = institution_annual_costs(output_db, institution_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = institution_power_generated(output_db, institution_id, truncate=False)
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
		
def institution_power_generated(output_db, institution_id, truncate=True):
	"""Input : sqlite output database and institution agent id
	Output : Sum of electricity generated in MWh every years in the institution reactor fleet
	"""
	db = dbopen(output_db)
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

def institution_lcoe(output_db, institution_id):
	"""Input : sqlite output database and institution agent id
	Output : Value corresponding to Levelized Cost of Electricity ($/MWh)
	"""
	db = dbopen(output_db)
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
	costs = institution_annual_costs(output_db, institution_id, truncate=False)
	costs['TotalCosts'] = costs.sum(axis=1)
	commissioning = costs['Capital'].idxmax()
	costs['Power'] = institution_power_generated(output_db, institution_id)
	costs = costs.fillna(0)
	power_generated = 0
	total_costs = 0
	for i in costs.index:
		power_generated += costs['Power'][i] / ((1 + default_discount_rate) ** (i - commissioning))
		total_costs += costs['TotalCosts'][i] / ((1 + default_discount_rate) ** (i - commissioning))
	return total_costs / power_generated

def institution_average_lcoe(output_db, institution_id):
	"""Input : sqlite output database and institution agent id
	Output : Variable cost corresponding at each time step (i.e. every year) to the weighted average of the reactors Levelized Cost of Electricity ($/MWh). A reactor is taken into account at time step t only if it is in activity (i.e. already commissioned and not yet decommissioned) at time step t.
	"""
	db = dbopen(output_db)
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
	dfEntry = dfEntry[dfEntry.ParentId==institution_id]
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
		tmp = lcoe(output_db, id)
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

def region_annual_costs(output_db, region_id, capital=True, truncate=True):
	"""Input : sqlite output database and region's AgentId. It is possible not to take into account the construction costs (capital=False) if the reactors are supposed to have been built before the beginning of the simulation. It is also possible to truncate the simulation results and only have access to cash flows occurring between the two dates (begin and end) specified in 'parameters.xml'. The truncation allows to let reactors decommission after the end of the simulation and thus to take into account cash flows that occur after the end of the simulation for example to calculate the LCOE.
	Output : total reactor costs per year over its lifetime at the region level.
	"""
	db = dbopen(output_db)
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
		
def region_annual_costs_present_value(output_db, region_id, capital=True, truncate=True):
	"""Same as annual_cost except all values are actualized to the begin date of the SIMULATION
	"""
	costs = region_annual_costs(output_db, region_id, capital)
	actualization = actualization_vector(len(costs))
	actualization.index = costs.index
	return costs.apply(lambda x : x * actualization)
		
def region_cumulative_capital(output_db, region_id):
	"""Input : sqlite output database and region agent id
	Output : cumulative sum of income and expense (= - expenditures + income)
	"""
	costs = - region_annual_costs(output_db, region_id).sum(axis=1)
	power_gen = region_power_generated(output_db, region_id) * region_average_lcoe(output_db, region_id)['Average LCOE']
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
		
def region_period_costs(output_db, region_id, t0=0, period=20, capital=True):
	"""Input : sqlite output database, region id, time window (t0, period) 
	Output : cost at each time step t corresponding to actualized sum of expense in [t+t0, t+t0+period] divided by actualized power generated in [t+t0, t+t0+period]
	"""
	db = dbopen(output_db)
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
	costs = region_annual_costs(output_db, region_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = region_power_generated(output_db, region_id, truncate=False)
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
	
def region_period_costs2(output_db, region_id, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
	"""
	db = dbopen(output_db)
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
	costs = region_annual_costs(output_db, region_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = region_power_generated(output_db, region_id, truncate=False)
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
		
def region_power_generated(output_db, region_id, truncate=True):
	"""Input : sqlite output database and region agent id
	Output : Sum of electricity generated in MWh every years in the region reactor fleet
	"""
	db = dbopen(output_db)
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

def region_lcoe(output_db, region_id):
	"""Input : sqlite output database and region agent id
	Output : Value corresponding to Levelized Cost of Electricity ($/MWh)
	"""
	db = dbopen(output_db)
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
	costs = region_annual_costs(output_db, region_id, truncate=False)
	costs['TotalCosts'] = costs.sum(axis=1)
	commissioning = costs['Capital'].idxmax()
	costs['Power'] = region_power_generated(output_db, region_id)
	costs = costs.fillna(0)
	power_generated = 0
	total_costs = 0
	for i in costs.index:
		power_generated += costs['Power'][i] / ((1 + default_discount_rate) ** (i - commissioning))
		total_costs += costs['TotalCosts'][i] / ((1 + default_discount_rate) ** (i - commissioning))
	return total_costs / power_generated

def region_average_lcoe(output_db, region_id):
	"""Input : sqlite output database and region agent id
	Output : Variable cost corresponding at each time step (i.e. every year) to the weighted average of the reactors Levelized Cost of Electricity ($/MWh). A reactor is taken into account at time step t if and only if it is in activity (i.e. already commissioned and not yet decommissioned) at time step t.
	"""
	db = dbopen(output_db)
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
		tmp = lcoe(output_db, id)
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

def simulation_annual_costs(output_db, capital=True, truncate=True):
	"""Input : sqlite output database. It is possible not to take into account the construction costs (capital=False) if the reactors are supposed to have been built before the beginning of the simulation. It is also possible to truncate the simulation results and only have access to cash flows occurring between the two dates (begin and end) specified in 'parameters.xml'. The truncation allows to let reactors decommission after the end of the simulation and thus to take into account cash flows that occur after the end of the simulation for example to calculate the LCOE.
	Output : total reactor costs per year over its lifetime at the simulation level.
	"""
	db = dbopen(output_db)
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
		
def simulation_annual_costs_present_value(output_db, capital=True, truncate=True):
	"""Same as annual_cost except all values are actualized to the begin date of the SIMULATION
	"""
	df = simulation_annual_costs(output_db, capital, truncate)
	for year in df.index:
		df.loc[year, :] = df.loc[year, :] / (1 + default_discount_rate) ** (year - df.index[0])
	return df
	
def simulation_cumulative_capital(output_db):
	"""Input : sqlite output database
	Output : cumulative sum of total income and total expense (= - expenditures + income) when all reactors of the simulation are taken into account
	"""
	costs = - simulation_annual_costs(output_db).sum(axis=1)
	power_gen = simulation_power_generated(output_db) * simulation_average_lcoe(output_db)['Average LCOE']
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
		
def simulation_period_costs(output_db, t0=0, period=20, capital=True):
	"""Input : sqlite output database, time window (t0, period) 
	Output : cost at each time step t corresponding to actualized sum of total expense in [t+t0, t+t0+period] divided by total actualized power generated in [t+t0, t+t0+period] when all reactors of the simulation are taken into account
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	dfEcoInfo = evaler.eval('EconomicInfo')
	simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
	simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
	costs = simulation_annual_costs(output_db, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = simulation_power_generated(output_db, truncate=False)
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
	
def simulation_period_costs2(output_db, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	dfEcoInfo = evaler.eval('EconomicInfo')
	simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
	simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
	costs = simulation_annual_costs(output_db, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = simulation_power_generated(output_db, truncate=False)
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
		
def simulation_power_generated(output_db, truncate=True):
	"""Input : sqlite output database
	Output : Electricity generated in MWh every years by all the reactors of the simulation
	"""
	db = dbopen(output_db)
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

def simulation_lcoe(output_db):
	"""Input : sqlite output database
	Output : Value corresponding to Levelized Cost of Electricity ($/MWh) when taking into account all reactors commissioned in the simulation
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db, write=False)
	dfInfo = evaler.eval('Info').reset_index()
	duration = dfInfo.loc[0, 'Duration']
	initialYear = dfInfo.loc[0, 'InitialYear']
	initialMonth = dfInfo.loc[0, 'InitialMonth']
	dfEcoInfo = evaler.eval('EconomicInfo')
	simulationBegin = dfEcoInfo[('Truncation', 'Begin')].iloc[0]
	simulationEnd = dfEcoInfo[('Truncation', 'End')].iloc[0]
	costs = simulation_annual_costs(output_db, truncate=False)
	costs['TotalCosts'] = costs.sum(axis=1)
	commissioning = costs['Capital'].idxmax()
	costs['Power'] = simulation_power_generated(output_db)
	costs = costs.fillna(0)
	power_generated = 0
	total_costs = 0
	for i in costs.index:
		power_generated += costs['Power'][i] / ((1 + default_discount_rate) ** (i - commissioning))
		total_costs += costs['TotalCosts'][i] / ((1 + default_discount_rate) ** (i - commissioning))
	return total_costs / power_generated

def simulation_average_lcoe(output_db):
	"""Input : sqlite output database and region agent id
	Output : Variable cost corresponding at each time step (i.e. every year) to the weighted average of the reactors Levelized Cost of Electricity ($/MWh). A reactor is taken into account at time step t if and only if it is in activity (i.e. already commissioned and not yet decommissioned) at time step t.
	"""
	db = dbopen(output_db)
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
		tmp = lcoe(output_db, id)
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

###########################
# Plotting costs #
###########################

# Reactor level

def annual_costs_plot(output_db, reactor_id, capital=True):
    """Plot of total costs for one reactor per year
    """
    df = annual_costs(output_db, reactor_id, capital) / 10 ** 9
    df.plot(kind='area')
    plt.xlabel('Year')
    plt.ylabel('Cost (billion $2015)')
    plt.title('Annual costs for nuclear plants over their lifetime')
    plt.show()
    
def annual_costs_present_value_plot(output_db, reactor_id, capital=True):
    """Plot of total costs for one reactor per year
    """
    df = annual_costs_present_value(output_db, reactor_id, capital) / 10 ** 9
    df.plot(kind='area')
    plt.xlabel('Year')
    plt.ylabel('Cost (billion $2015)')
    plt.title('Annual costs for nuclear (present value)')
    plt.show()

def average_cost_plot(output_db, reactor_id, capital=True):
    """Plot of the average costs for one reactor over its lifetime
    """
    if not isinstance(reactor_id, list):
    	raise TypeError('Wrong input, reactor ids should be given in a list')
    db = dbopen(output_db)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo['Duration'].iloc[0]
    df = pd.DataFrame(index=list(range(duration)))
    initialYear = dfInfo['InitialYear'].iloc[0]
    initialMonth = dfInfo['InitialMonth'].iloc[0]
    dfEntry = evaler.eval('AgentEntry').reset_index()
    for id in reactor_id:
    	dfEntry2 = dfEntry[dfEntry.AgentId==id]
    	date_entry = dfEntry2['EnterTime'].iloc[0]
    	lifetime = dfEntry2['Lifetime'].iloc[0]
    	prototype = dfEntry2['Prototype'].iloc[0]
    	ser = pd.Series(average_cost(output_db, id, capital), index=list(range(date_entry, date_entry + lifetime)))
    	df[prototype+' (AgentId : '+str(id)+')'] = ser
    df = df.fillna(0)
    df['Date'] = pd.Series(df.index.values).apply(lambda x: (x + initialMonth - 1) // 12 + initialYear + (x % 12) / 12)
    df = df.set_index('Date')
    df.plot()
    plt.xlabel('Year')
    plt.ylabel('Cost ($2015/MWh)')
    plt.title('Average costs for nuclear plants over their lifetime')
    plt.show()
    	
def lcoe_plot(output_db, reactor_id, capital=True):
    """Plot of levelized cost of electricity obtained with one reactor (given a
    fuel cycle technology)
    """
    if not isinstance(reactor_id, list):
    	raise TypeError('Wrong input, reactor ids should be given in a list')
    db = dbopen(output_db)
    evaler = Evaluator(db, write=False)
    dfInfo = evaler.eval('Info').reset_index()
    duration = dfInfo['Duration'].iloc[0]
    initialYear = dfInfo['InitialYear'].iloc[0]
    initialMonth = dfInfo['InitialMonth'].iloc[0]
    dfEntry = evaler.eval('AgentEntry').reset_index()
    df = pd.DataFrame(index=list(range(duration)))
    for id in reactor_id:
    	dfEntry2 = dfEntry[dfEntry.AgentId==id]
    	date_entry = dfEntry2['EnterTime'].iloc[0]
    	lifetime = dfEntry2['Lifetime'].iloc[0]
    	prototype = dfEntry2['Prototype'].iloc[0]
    	ser = pd.Series(lcoe(output_db, id, capital), index=list(range(date_entry,lifetime)))
    	df[prototype+' (AgentId : '+str(id)+')'] = ser
    df = df.fillna(0)
    df['Date'] = pd.Series(df.index.values).apply(lambda x: (x + initialMonth - 1) // 12 + initialYear + (x % 12) / 12)
    df = df.set_index('Date')
    df.plot()
    #plt.plot(df)#, label=prototype+' (AgentId : '+str(reactor_id)+')')
    #legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.xlabel('Year')
    plt.ylabel('Cost ($2015/MWh)')
    plt.title('Levelized cost of electricity for nuclar plants')
    plt.show()

# Institution level

def institution_annual_costs_plot(output_db, institution_id, capital=True):
	"""plot all reactors annual costs for a given institution
	"""
	total = institution_annual_costs(output_db, institution_id, capital) / 10 ** 9 # billion $2015
	total.plot(kind='area', colormap='Greens', linewidth = 3)
	plt.xlabel('Year')
	plt.ylabel('Cost (billion $2015)')
	plt.title('Annual costs related to its nuclear plants for institution ' + str(institution_id))
	plt.show()
	
def institution_annual_costs_present_value_plot(output_db, institution_id, capital=True):
	"""plot all reactors annual costs for a given institution
	"""
	total = institution_annual_costs_present_value(output_db, institution_id, capital) / 10 ** 9 # billion $2015
	total.plot(kind='area', colormap='Greens', linewidth = 3)
	plt.xlabel('Year')
	plt.ylabel('Cost (billion $2015)')
	plt.title('Annual costs related to its nuclear plants for institution ' + str(institution_id) + ' (present value)')
	plt.show()

def institution_period_costs_plot(output_db, institution_id, period=20, capital=True):
	"""New manner to calculate price of electricity, maybe more accurate than lcoe : calculate all costs in a n years period and then determine how much the cost of electricity should be at an institutional level
	"""
	institution_period_costs(output_db, institution_id, period, capital)['Ratio'].plot()
	plt.xlabel('Year')
	plt.ylabel('Cost ($2015/MWh)')
	plt.title('Reactor costs using a ' + str(period) + ' years time frame (institution n°' + str(institution_id)+ ')')
	plt.show()

# Region level

def region_annual_costs_plot(output_db, region_id, capital=True):
	"""plot all reactors annual costs for a given institution
	"""
	total = region_annual_costs(output_db, region_id, capital) / 10 ** 9
	total.plot(kind='area', colormap='Blues', linewidth = 3)
	plt.xlabel('Year')
	plt.ylabel('Cost (billion $2015)')
	plt.title('Annual costs related to its nuclear plants for region ' + str(region_id))
	plt.show()
	
def region_annual_costs_present_value_plot(output_db, region_id, capital=True):
	"""plot all reactors annual costs for a given institution
	"""
	total = region_annual_costs_present_value(output_db, region_id, capital) / 10 ** 9
	total.plot(kind='area', colormap='Blues', linewidth = 3)
	plt.xlabel('Year')
	plt.ylabel('Cost (billion $2015)')
	plt.title('Annual costs related to its nuclear plants for region ' + str(region_id) + ' (present value)')
	plt.show()

def region_period_costs_plot(output_db, region_id, period=20, capital=True):
	"""Same as instution_period_costs but at a regional level
	"""
	region_period_costs(output_db, region_id, period)['Ratio'].plot()
	plt.xlabel('Year')
	plt.ylabel('Cost ($2015/MWh)')
	plt.title('Reactor costs using a ' + str(period) + ' years time frame (region n°' + str(region_id)+ ')')
	plt.show()
	