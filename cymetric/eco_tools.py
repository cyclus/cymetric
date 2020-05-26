"""This file stores several tools that are very useful for the economic
analysis (e.g. for eco_metrics.py) as it adds complexity and most importantly
realism. First some financial parameters are taken from papers about nuclear
power economics. Second, some functions give several modeling options in the
metrics calculations. All prices are 2015 $
"""

import pandas as pd
import numpy as np
import yaml
import math
from cymetric.tools import ensure_dt_bytes

eco_props_agents = ["capital",
                    "operation_maintenance",
                    "decommission",
                    "fuels"]
eco_props_region = ["finance"] + eco_props_agents
eco_properties = ["periode"] + eco_props_region

finance_col = ["discount_rate", "tax_rate", "return_on_debt",
               "return_on_equity"]
capital_col = ["beforePeak", "afterPeak", "constructionDuration",
               "overnight_cost", "capital_dev"]
operation_col = ["fixed", "variable", "operation_dev"]
fuel_col = ["name", "supply_cost", "waste_fee", "fuel_dev"]


class eco_input_data():
    """The EconomicInfo metric stores all economic data needed to calculate the
    economic metrics. These economic parameters are originally written in
    'parameters.xml'.
    """

    def __init__(self, filename):
        self.load_economic_info(filename)

    def load_economic_info(self, eco_input):
        stream = open(eco_input)
        data = yaml.load(stream, Loader=yaml.FullLoader)
        self.dict = data

    def update_prop(self, proto, traveling_dict, key, properties, prop_list):
        updated = False
        if key in traveling_dict:
            for agent in traveling_dict[key]:
                if agent["prototype"] == proto:
                    traveling_dict = agent
                    updated = True
                    for prop in prop_list:
                        if prop in traveling_dict:
                            properties[prop] = traveling_dict[prop]
        return updated, traveling_dict

    def get_prototype_eco(self, filiation):
        proto_eco = {}
        traveling_dict = self.dict["eco_model"]
        for prop in eco_properties:
            if prop in traveling_dict:
                proto_eco[prop] = traveling_dict[prop]

        for proto in filiation:
            updated = False
            status, traveling_dict = self.update_prop(proto, traveling_dict,
                                                      "region", proto_eco,
                                                      eco_props_region)
            if not updated:
                status, traveling_dict = self.update_prop(proto,
                                                          traveling_dict,
                                                          "institution",
                                                          proto_eco,
                                                          eco_props_agents)
            if not updated:
                status, traveling_dict = self.update_prop(proto,
                                                          traveling_dict,
                                                          "facility",
                                                          proto_eco,
                                                          eco_props_agents)

        return proto_eco

    def get_prototypes_eco(self):
        proto_eco = {}
        model_dict = self.dict["eco_model"]

        for prop in eco_properties:
            if prop in model_dict:
                proto_eco[prop] = model_dict[prop]

        for region_dict in model_dict["region"]:
            region_eco = proto_eco
            region_eco["prototype"] = region_dict["prototype"]
            for prop in eco_properties:
                if prop in region_dict:
                    proto_eco[prop] = region_dict[prop]
            print(region_eco)
            region_raw = build_eco_row(region_eco)
            print(region_raw)

        return proto_eco

finance_col = ["discount_rate", "tax_rate", "return_on_debt",
               "return_on_equity"]
capital_col = ["beforePeak", "afterPeak", "constructionDuration",
               "overnight_cost", "capital_dev"]
operation_col = ["fixed", "variable", "operation_dev"]
fuel_col = ["name", "supply_cost", "waste_fee", "fuel_dev"]

def build_eco_row(proto_dict):
    row_col = finance_col + capital_col + operation_col + fuel_col
    print(row_col)
    df = pd.DataFrame(columns=row_col)
    

    for fuel_type in proto_dict["fuels"]:
        a_row = pd.DataFrame(np.array([(
                    float(proto_dict["finance"]["discount_rate"]),
                    float(proto_dict["finance"]["tax_rate"]),
                    float(proto_dict["finance"]["return_on_debt"]),
                    float(proto_dict["finance"]["return_on_equity"]),
                    float(proto_dict["capital"]["beforePeak"]),
                    float(proto_dict["capital"]["afterPeak"]),
                    float(proto_dict["capital"]["constructionDuration"]),
                    float(proto_dict["capital"]["overnight_cost"]),
                    float(proto_dict["capital"]["deviation"]),
                    float(proto_dict["operation_maintenance"]["fixed"]),
                    float(proto_dict["operation_maintenance"]["variable"]),
                    float(proto_dict["operation_maintenance"]["deviation"]),
                    fuel_type["name"],
                    float(fuel_type["supply_cost"]),
                    float(fuel_type["waste_fee"]),
                    float(fuel_type["deviation"]))
                    ],
                    dtype=ensure_dt_bytes([
                         ('discount_rate', '<f8'),
                         ('tax_rate', '<f8'),
                         ('return_on_debt', '<f8'),
                         ('return_on_equity', '<f8'),
                         ('beforePeak', '<f8'),
                         ('afterPeak', '<f8'),
                         ('constructionDuration', '<f8'),
                         ('overnight_cost', '<f8'),
                         ('operation_dev', '<f8'),
                         ('fixed', '<f8'),
                         ('variable', '<f8'),
                         ('capital_dev', '<f8'),
                         ('name', 'O'),
                         ('supply_cost', '<f8'),
                         ('waste_fee', '<f8'),
                         ('fuel_dev', '<f8')
                         ])))
        print(a_row)
        df = pd.concat([a_row, df], ignore_index=True)
    return df


def get_filiation_per_name(name, dfEntry):
    filiation = []
    filiation.append(name)
    parent_id = dfEntry[dfEntry["Prototype"] == name]["ParentId"][0]

    while parent_id != -1:
        # get the parent
        parent_entry = dfEntry[dfEntry["AgentId"] == parent_id]
        parent_name = parent_entry.iloc[0]["Prototype"]
        filiation.append(parent_name)
        # get the parent of the parent
        parent_id = parent_entry.iloc[0]["ParentId"]
    return list(reversed(filiation))


def get_filiation_per_id(id, dfEntry):
    name = dfEntry[dfEntry["AgentId"] == id]["Prototype"][0]
    return get_filiation_per_name(name, dfEntry)


####################
# Mining & Milling #
####################


def isuraniumsource(dfEntry, id):
    """
    Input : Agents entry table and agent id
    Output : boolean (True if facility is a mining facility, False if not).
    """
    return 'SOURCE' in dfEntry[dfEntry.AgentId == id]['Spec'].iloc[0].upper()


##############
# Conversion #
##############


def isconversionplant(dfEntry, id):
    """
    Input : Agent entry table and agent id
    Output : boolean (True if facility is a conversion plant, False if not).
    """
    return 'CONV' in dfEntry[dfEntry.AgentId == id]['Spec'].iloc[0].upper()

##############
# Enrichment #
##############


def swu(feedMass, feedAssay, productMass, productAssay, wasteMass, wasteAssay):
    """
    Input : mass and assay of feed, product and waste
    Output : corresponding amount of swu
    """
    rtn_value = wasteMass * V(wasteAssay) + productMass * V(productAssay) \
        - feedMass * V(feedAssay)


def waste_mass(feedMass, productMass):
    """
    Input : feed and product masses
    Output : waste mass
    """
    return feedMass - productMass


def waste_assay(feedMass, feedAssay, productMass, productAssay, wasteMass):
    """
    Input : mass of feed, product and waste, assay of feed and product
    Output : waste assay
    """
    return (feedMass * feedAssay - productMass * productAssay) / wasteMass


def V(x):
    """Value function used to calculate the swu
    """
    return (2 * x - 1) * math.log(x / (1 - x))


def isenrichmentplant(dfEntry, id):
    """
    Input : Agents entry table and agent id
    Output : boolean (True if facility is an enrichment plant, False if not).
    """
    return 'ENRICH' in dfEntry[dfEntry.AgentId == id]['Spec'].iloc[0].upper()

###############
# Fabrication #
###############


def isfuelfab(dfEntry, id):
    """
    Input : Agents entry table and agent id
    Output : boolean (True if facility is a fuel fabrication plant).
    """
    return 'FAB' in dfEntry[dfEntry.AgentId == id]['Spec'].iloc[0].upper()

###########
# Reactor #
###########


def capital_shape(beforePeak=48, afterPeak=48):
    """
    Input : relative position of to the peak.two parameters defining the size
    of the shape
    Output : curve with integral equals to one in the requested shape.
    """
    if (not isinstance(beforePeak, int)) or (not isinstance(afterPeak, int)):
        raise Exception("input parameters must be integers")
    step1 = pd.Series(list(range(beforePeak)))
    step1 = step1.apply(lambda x:
                        2 * x / (beforePeak * (beforePeak + afterPeak)))

    step2 = pd.Series(list(range(beforePeak, beforePeak + afterPeak + 1)))
    step2 = step2.apply(lambda x:
                        -2 / (afterPeak * (beforePeak + afterPeak))
                        * (x - beforePeak - afterPeak))

    return pd.concat([step1, step2]).reset_index()[0]


def discount_rate(amountOfDebt, amountOfEquity, taxRate,
                  returnOnDebt, returnOnEquity, inflationRate):
    """
    Input : share of debt, share of equity, tax rate, return on debt, return on
    equity and inflation rate
    Output : corresponding discount rate
    source: D'Haeseleer p.81
    """
    nominalRate = returnOnDebt * amountOfDebt + returnOnEquity * amountOfEquity
    realRate = (1 + nominalRate) / (1 + inflationRate) - 1


def overnight_cost(foak, n):
    """
    Input : price of First Of A Kind reactor
    Output : price of n-th Of A Kind reactor
    source :
    http://www.power-eng.com/content/dam/pe/online-articles/documents/2011/july/EPRI.pdf
    https://www.netl.doe.gov/File%20Library/research/energy%20analysis/publications/QGESS_FOAKtoNOAK_Final.pdf
    http://www.rff.org/events/documents/rffexperiencecurvetalk.pdf
    LR~20% => b=
    """
    b = 0.002888279324826512
    return foak * n ** (-b)


def substitution_power_purchase(annualCosts, power, substitutePrice,
                                yearBegin, yearEnd):
    """
    Input : annual costs (Construction costs + fuel costs + O&M +
    decommissioning), substitute power needed (MWh), price of substitute power
    ($/MWh), interval of time when the substitute power is needed ([yearBegin,
    yearEnd[)
    Output : annual costs with substitution power
    """
    if 'Substitute' in annualCosts.columns:
        annualCosts.loc[yearBegin:yearEnd,
                        'Substitute'] += substitutePrice * power
    else:
        annualCosts['Substitute'] = pd.Series()
        annualCosts = annualCosts.fillna(0)
        annualCosts.loc[yearBegin:yearEnd,
                        'Substitute'] += substitutePrice * power


def isreactor(dfPower, id):
    """
    Input : reactor agent id and pandas DataFrame with power generated. Agent
    generates power if and only if it is a reactor
    Output : boolean (True if agent id corresponds to a reactor, False if not)
    """
    return not dfPower[dfPower.AgentId == id].empty

##############
# Separation #
##############


def isseparation(dfEntry, id):
    """Input : Agents entry table and agent id
    Output : boolean (True if facility is a separation plant, False if not).
    """
    return 'SEP' in dfEntry[dfEntry.AgentId == id]['Spec'].iloc[0].upper()


#######################
# Price actualization #
#######################

def actualization_vector(size, discountRate):
    """Output : pandas Series with actualization factors
    """
    rtn = pd.Series(1 / (1 + discountRate), index=list(range(size))).cumprod()
    return rtn * (1 + discountRate)


def actualize(price, delta_t, discount_rate):
    """Given a price at date t + delta_t, give the actualized price at t.
    """
    return price / (1 + discount_rate) ** delta_t


def inflation(price, date):
    """Give the 2015 $ value of a price given in 'date' $
    """
    return price * f_inflation.loc[date, 0]
