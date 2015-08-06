"""This file stores several tools that are very useful for the economic analysis (e.g. for eco_metrics.py) as it adds complexity and most importantly realism. First some financial parameters are taken from papers about nuclear power economics. Second, some functions give several modeling options in the metrics calculations.
All prices are 2015 $
"""

import pandas as pd
import numpy as np
import math

################################
# Default Financial Parameters #
################################

# see d'Haeseleer (2013 data)

# Region level
default_discount_rate = 0.05
#tax_rate = # %
#depreciation_schedule = # depreciation type
#depreciation_time = # years
#external_cost = # $/MWh (e.g. CO2)
#CO2_price = # $/tonCO2
#ini_BC = # $/MWh initial busbar cost
#delta_BC = # $/MWh increment of the BC in the loop

# Institution level
#fixedOM = # $/MWh-yr
#variableOM = # $/MWh
#default_fixedOM
#default_variableOM
defaultOM = 10 # $/MWh

# Facility level
#overnight = 5000 # $/MW cap or maybe overnight cap cost
#default_construction_time = # years
#decommissioning_cost = # $/MW
#default_decommissioning_time = # years


dct={
1970: (1 + 10**(-2) *	5.84),
1971: (1 + 10**(-2) *	4.30),
1972: (1 + 10**(-2) *	3.27),
1973: (1 + 10**(-2) *	6.16),
1974: (1 + 10**(-2) *	11.03),
1975: (1 + 10**(-2) *	9.20),
1976: (1 + 10**(-2) *	5.75),
1977: (1 + 10**(-2) *	6.50),
1978: (1 + 10**(-2) *	7.62),
1979: (1 + 10**(-2) *	11.22),
1980: (1 + 10**(-2) *	13.58),
1981: (1 + 10**(-2) *	10.35),
1982: (1 + 10**(-2) *	6.16),
1983: (1 + 10**(-2) *	3.22),
1984: (1 + 10**(-2) *	4.30),
1985: (1 + 10**(-2) *	3.55),
1986: (1 + 10**(-2) *	1.91),
1987: (1 + 10**(-2) *	3.66),
1988: (1 + 10**(-2) *	4.08),
1989: (1 + 10**(-2) *	4.83),
1990: (1 + 10**(-2) *	5.39),
1991: (1 + 10**(-2) *	4.25),
1992: (1 + 10**(-2) *	3.03),
1993: (1 + 10**(-2) *	2.96),
1994: (1 + 10**(-2) *	2.61),
1995: (1 + 10**(-2) *	2.81),
1996: (1 + 10**(-2) *	2.93),
1997: (1 + 10**(-2) *	2.34),
1998: (1 + 10**(-2) *	1.55),
1999: (1 + 10**(-2) *	2.19),
2000: (1 + 10**(-2) *	3.38),
2001: (1 + 10**(-2) *	2.83),
2002: (1 + 10**(-2) *	1.59),
2003: (1 + 10**(-2) *	2.27),
2004: (1 + 10**(-2) *	2.68),
2005: (1 + 10**(-2) *	3.39),
2006: (1 + 10**(-2) *	3.24),
2007: (1 + 10**(-2) *	2.85),
2008: (1 + 10**(-2) *	3.85),
2009: (1 + 10**(-2) *	-0.34),
2010: (1 + 10**(-2) *	1.64),
2011: (1 + 10**(-2) *	3.16),
2012: (1 + 10**(-2) *	2.07),
2013: (1 + 10**(-2) *	1.47),
2014: (1 + 10**(-2) *	1.62),
2015: 1
}

f_inflation = pd.DataFrame(dct, index=[0]).transpose()

for year in reversed(f_inflation.index[:-1]):
	f_inflation.loc[year, 0] *= f_inflation.loc[year + 1, 0]
	
	
# 2007 $ "The Future of Nuclear Fuel Cycle" (page 102)
dct_fuel = {'Natural Uranium ($/kgHM)' : 80, 'Depleted Uranium ($/kgHM)' : 10, 'Conversion of Natural U ($/kgHM)' : 10, 'Enrichment of Natural U ($/SWU)' : 160, 'Fabrication of UOX from Natural U ($/kgHM)' : 250, 'Conversion of Repr. U' : 0.02, 'Enrichment of Repr. U' : 0.1, 'Fabrication of UOX from Repr U' : 0.07, 'Fabrication of MOX ($/kgHM)' : 2400, 'Fabrication of FR fuel ($/kgHM)' : 2400, 'LWR Capital (overnight, $/kWe)' : 4000, 'LWR Capacity Factor' : 0.85, 'FR Capital premium' : 0.2, 'FR O&M premium' : 0.2, 'FR Capacity Factor' : 0.85, 'UOX, PUREX ($/kgHM)': 1600, 'FR fuel, pyroprocessing ($/kgHM)' : 3200, 'Interim Storage of UOX ($/kgiHM)' : 200, 'Interim Storage of MOX ($/kgiHM)' : 200, 'Disposal of Spent UOX ($/kgiHM)' : 470, 'Disposal of Spent MOX ($/kgiHM)' : 3130, 'Disposal of HLW from UOX (PUREX) ($/kgiHM, d. factor, $/kgFP)' : [190, 2.5, 3650], 'Disposal of HLW from UOX (TRUEX) ($/kgiHM)' : 190, 'Disposal of HLW from FR ($/kgiHM)' : 280}


######################
# Default Fuel costs #
######################

# see d'Haeseleer (2013 data)

# mining
#u_ore_price = 
# processing
#yellow_cake_price = 130 # $/kg
# conversion from U308 to UF6
#conversion_cost = 9 # $/kg
# enrichment
#swu_cost = 140 # $/swu
# Example : for 4.95 % enrichment, fuel cost 300 $/kg
# fabrication
## reconversion
#UO2_cost =
## fuel reprocessing
#Pu_price =
#reprocessed_ur =
# dict with prices for different reactors (pwr, phwr, bwr, fr... differentiate 
# reenriched uox with uox using natural uranium)
#fuel_price = {} 
# a function could give the price of uranium as a function of availability (see 
# Arnaud's work)
# "The back-end cost elements include the interim storage facilities, 
# construction of reprocessing facilities, SNF encapsulation and final 
# disposal" (d'Haeseleer)

default_fuel_price = 2500 # $/kg, when institution doesn't generate power


######################
# Capital costs data #
######################

rapid_cap_begin = 36
# idea : take real durations from real examples and quote them
rapid_cap_duration = 72
slow_cap_begin = 103 #84
slow_cap_duration = 175#144
default_cap_begin = 60
default_cap_duration = 108
default_cap_overnight = 4000000 # $/MW
default_cap_shape = "triangle"
"""
Distinguishing between twin/single (still for NOAK2(5+) ) leads to:
= about 3230 EUR2012/kW for NOAK2 (5+) with uncertainty -10 to + 15%
on a brownfield, for a twin unit
= about 3570 EUR2012/kW for NOAK2 (5+) with uncertainty -10 to + 15%
on a brownfield, for a single unit
For a first of a kind in a particular country (FOAK2), we obtain:
= about 3910 EUR2012/kW for FOAK2 with uncertainty -20 to + 30%
on a brownfield, for a twin unit
= about 4250 EUR2012/kW for FOAK2 with uncertainty -20 to + 30%
on a brownfield, for a single unit
"""
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

shapes = ["triangle", "rectangle", "plateau"]

def capital_shape(t0=5, duration=10, shape='triangle'):
    """Input : two parameters defining the size of the shape
    Output : curve with integral equals to one in the requested shape.
    """
    if not isinstance(t0, int):
        raise Exception("Year begin for paiement must be an integer")
    if t0 < 0:
        raise Exception("Year begin for paiement must be positive")
    if not isinstance(duration, int):
        raise Exception("Duration of paiement must be an integer")
    if duration < 0:
        raise Exception("Duration of paiement must be positive")            
    if "TRIANGLE" in shape.upper():
        step1 = pd.Series(list(range(t0))).apply(lambda x: 2/(t0*duration)*x)
        step2 = pd.Series(list(range(duration-t0+1))).apply(lambda x: -2/((
              duration-t0)*duration)*x+2/duration)
        return pd.concat([step1, step2]).reset_index()[0]      
    elif "RECTANGLE" in shape.upper():
        return pd.Series([1/duration]*duration)
    else:
        raise Exception("Wrong shape, valid shapes are in the following list : " + str(shapes))
        
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
	
def isreactor(id, dfPower):
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

def actualization_vector(size):
	"""Output : pandas Series with actualization factors
	"""
	rtn = pd.Series(1 / (1 + default_discount_rate), index=list(range(size))).cumprod()
	return rtn * (1 + default_discount_rate)

def actualize(price, delta_t, discount_rate=default_discount_rate):
    """Given a price at date t + delta_t, give the actualized price at t.
    """
    return price / (1 + discount_rate) ** delta_t
    
def inflation(price, date):
	"""Give the 2015 $ value of a price given in 'date' $
	"""
	return price * f_inflation.loc[date, 0]