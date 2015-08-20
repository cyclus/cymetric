"""This file stores several tools that are very useful for the economic analysis (e.g. for eco_metrics.py) as it adds complexity and most importantly realism. First some financial parameters are taken from papers about nuclear power economics. Second, some functions give several modeling options in the metrics calculations.
All prices are 2015 $
"""

import pandas as pd
import numpy as np
import math


####################
# Mining & Milling #
####################

def isuraniumsource(dfEntry, id):
	"""Input : Agents entry table and agent id
	Output : boolean (True if facility is a mining facility, False if not).
	"""
	return 'SOURCE' in dfEntry[dfEntry.AgentId==id]['Spec'].iloc[0].upper()

##############
# Conversion #
##############

def isconversionplant(dfEntry, id):
	"""Input : Agent entry table and agent id
	Output : boolean (True if facility is a conversion plant, False if not).
	"""
	return 'CONV' in dfEntry[dfEntry.AgentId==id]['Spec'].iloc[0].upper()

##############
# Enrichment #
##############

def swu(feedMass, feedAssay, productMass, productAssay, wasteMass, wasteAssay):
	"""Input : mass and assay of feed, product and waste
	Output : corresponding amount of swu
	"""
	return wasteMass * V(wasteAssay) + productMass * V(productAssay) - feedMass * V(feedAssay)
	
def waste_mass(feedMass, productMass):
	"""Input : feed and product masses
	Output : waste mass
	"""
	return feedMass - productMass
	
def waste_assay(feedMass, feedAssay, productMass, productAssay, wasteMass):
	"""Input : mass of feed, product and waste, assay of feed and product
	Output : waste assay
	"""
	return (feedMass * feedAssay - productMass * productAssay) / wasteMass

def V(x):
	"""Value function used to calculate the swu
	"""
	return (2 * x - 1) * math.log(x / (1 - x))
	
def isenrichmentplant(dfEntry, id):
	"""Input : Agents entry table and agent id
	Output : boolean (True if facility is an enrichment plant, False if not).
	"""
	return 'ENRICH' in dfEntry[dfEntry.AgentId==id]['Spec'].iloc[0].upper()

###############
# Fabrication #
###############

def isfuelfab(dfEntry, id):
	"""Input : Agents entry table and agent id
	Output : boolean (True if facility is a fuel fabrication plant, False if not).
	"""
	return 'FAB' in dfEntry[dfEntry.AgentId==id]['Spec'].iloc[0].upper()

###########
# Reactor #
###########


def capital_shape(beforePeak=48, afterPeak=48):
    """Input : relative position of to the peak.two parameters defining the size of the shape
    Output : curve with integral equals to one in the requested shape.
    """
    if (not isinstance(beforePeak, int)) or (not isinstance(afterPeak, int)):
        raise Exception("input parameters must be integers")
    step1 = pd.Series(list(range(beforePeak))).apply(lambda x: 2/(beforePeak*(beforePeak+afterPeak))*x)
    step2 = pd.Series(list(range(beforePeak, beforePeak + afterPeak + 1))).apply(lambda x: -2/(afterPeak*(beforePeak+afterPeak))*(x-(beforePeak+afterPeak)))
    return pd.concat([step1, step2]).reset_index()[0]
    
        
def discount_rate(amountOfDebt, amountOfEquity, taxRate, returnOnDebt, returnOnEquity, inflationRate):
	"""Input : share of debt, share of equity, tax rate, return on debt, return on equity and inflation rate
	Output : corresponding discount rate
	D'Haeseleer p.81"""
	nominalRate = returnOnDebt * amountOfDebt + returnOnEquity * amountOfEquity
	realRate = (1 + nominalRate) / (1 + inflationRate) - 1
	
def overnight_cost(foak, n):
	"""Input : price of First Of A Kind reactor 
	Output : price of n-th Of A Kind reactor
	http://www.power-eng.com/content/dam/pe/online-articles/documents/2011/july/EPRI.pdf
	https://www.netl.doe.gov/File%20Library/research/energy%20analysis/publications/QGESS_FOAKtoNOAK_Final.pdf
	http://www.rff.org/events/documents/rffexperiencecurvetalk.pdf LR~20% => b="""
	b = 0.002888279324826512
	return foak * n ** (- b)
	
def substitution_power_purchase(annualCosts, power, substitutePrice, yearBegin, yearEnd):
	"""Input : annual costs (Construction costs + fuel costs + O&M + decommissioning), substitute power needed (MWh), price of substitute power ($/MWh), interval of time when the substitute power is needed ([yearBegin, yearEnd[)
	Output : annual costs with substitution power
	"""
	if 'Substitute' in annualCosts.columns:
		annualCosts.loc[yearBegin:yearEnd, 'Substitute'] += substitutePrice * power
	else:
		annualCosts['Substitute'] = pd.Series()
		annualCosts = annualCosts.fillna(0)
		annualCosts.loc[yearBegin:yearEnd, 'Substitute'] += substitutePrice * power
	
def isreactor(dfPower, id):
	"""Input : reactor agent id and pandas DataFrame with power generated. Agent generates power if and only if it is a reactor
	Output : boolean (True if agent id corresponds to a reactor, False if not)
	"""
	return not dfPower[dfPower.AgentId==id].empty

##############
# Separation #
##############

def isseparation(dfEntry, id):
	"""Input : Agents entry table and agent id
	Output : boolean (True if facility is a separation plant, False if not).
	"""
	return 'SEP' in dfEntry[dfEntry.AgentId==id]['Spec'].iloc[0].upper()


#######################
# Price actualization #
#######################

def actualization_vector(size, discountRate=default_discount_rate):
	"""Output : pandas Series with actualization factors
	"""
	rtn = pd.Series(1 / (1 + discountRate), index=list(range(size))).cumprod()
	return rtn * (1 + discountRate)

def actualize(price, delta_t, discount_rate=default_discount_rate):
    """Given a price at date t + delta_t, give the actualized price at t.
    """
    return price / (1 + discount_rate) ** delta_t
    
def inflation(price, date):
	"""Give the 2015 $ value of a price given in 'date' $
	"""
	return price * f_inflation.loc[date, 0]