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
        print("mydict", self.dict)


def load_economic_info(eco_input):
    tuples = ['Prototype',
              'AgentId',
              'ReturnOnDebt',
              'ReturnOnEquity',
              'TaxRate',
              'DiscountRate',
              'beforePeak',
              'afterPeak',
              'constructionDuration',
              'Captial_Deviation',
              'OvernightCost',
              'Duration',
              'OvernightCost',
              'FixedCost',
              'VariableCost',
              'OperationMaintenance_Deviation',
              'Commodity',
              'SupplyCost',
              'WasteFee',
              'Fuel_Deviation',
              'Begin',
              'End']
    #rtn = pd.DataFrame(index=tuples)
    #agentIndex = dfEntry.reset_index().set_index('AgentId')['index']
    #rtn = rtn.T
    #rtn['Agent_Prototype'] = dfEntry['Prototype']
    #rtn['Agent_AgentId'] = dfEntry['AgentId']
    #rtn['Agent_ParentId'] = dfEntry['ParentId']
    #parametersInput = 'tests/parameters.xml'
    #for child in root:
    #    print(child.tag, child.attrib)
    #
    #def add_finance(rtn, root):


    #def add_capital(rtn, root):
    #def add_decommisioning(rtn, root):

    #def add_operation_maintenance(rtn, root):

    #def add_fuel(rtn, root):
    #
    #add_finance(rtn, root)


    # for region in root.findall('region'):
    #    idRegion = int(region.find('id').text)
    #    finance = region.find('finance')
    #    if finance is not None:
    #        returnOnDebt = float(finance.find('return_on_debt').text)
    #        returnOnEquity = float(finance.find('return_on_equity').text)
    #        taxRate = float(finance.find('tax_rate').text)
    #        discountRate = float(finance.find('discount_rate').text)
    #        rtn.loc[agentIndex[idRegion], 'Finance_TaxRate'] = taxRate
    #        rtn.loc[agentIndex[idRegion], 'Finance_ReturnOnDebt'] = returnOnDebt
    #        rtn.loc[agentIndex[idRegion], 'Finance_ReturnOnEquity'] = returnOnEquity
    #        rtn.loc[gent_index[idRegion], 'Finance_DiscountRate'] = discountRate
    #        for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    #            rtn.loc[agentIndex[idInstitution], 'Finance_TaxRate'] = taxRate
    #            rtn.loc[agentIndex[idInstitution], 'Finance_ReturnOnDebt'] = returnOnDebt
    #            rtn.loc[agentIndex[idInstitution], 'Finance_ReturnOnEquity'] = returnOnEquity
    #            rtn.loc[gent_index[idInstitution], 'Finance_DiscountRate'] = discountRate
    #            for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    #                rtn.loc[agentIndex[idFacility], 'Finance_TaxRate'] = taxRate
    #                rtn.loc[agentIndex[idFacility], 'Finance_ReturnOnDebt'] = returnOnDebt
    #                rtn.loc[agentIndex[idFacility], 'Finance_ReturnOnEquity'] = returnOnEquity
    #                rtn.loc[gent_index[idFacility], 'Finance_DiscountRate'] = discountRate
    #    capital = region.find('capital')
    #    if capital is not None:
    #        beforePeak = int(capital.find('beforePeak').text)
    #        afterPeak = int(capital.find('afterPeak').text)
    #        constructionDuration = int(capital.find('constructionDuration').text)
    #        deviation = float(capital.find('deviation').text)
    #        overnightCost = float(capital.find('overnight_cost').text)
    #        rtn.loc[agentIndex[idRegion], 'Captial_beforePeak'] = beforePeak
    #        rtn.loc[agentIndex[idRegion], 'Captial_afterPeak'] = afterPeak
    #        rtn.loc[agentIndex[idRegion], 'Captial_constructionDuration'] = constructionDuration
    #        rtn.loc[agentIndex[idRegion], 'Captial_Deviation'] = deviation
    #        rtn.loc[agentIndex[idRegion], 'Captial_OvernightCost'] = overnightCost
    #        for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    #            rtn.loc[agentIndex[idInstitution], 'Captial_beforePeak'] = beforePeak
    #            rtn.loc[agentIndex[idInstitution], 'Captial_afterPeak'] = afterPeak
    #            rtn.loc[agentIndex[idInstitution], 'Captial_constructionDuration'] = constructionDuration
    #            rtn.loc[agentIndex[idInstitution], 'Captial_Deviation'] = deviation
    #            rtn.loc[agentIndex[idInstitution], 'Captial_OvernightCost'] = overnightCost
    #            for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    #                rtn.loc[agentIndex[idFacility], 'Captial_beforePeak'] = beforePeak
    #                rtn.loc[agentIndex[idFacility], 'Captial_afterPeak'] = afterPeak
    #                rtn.loc[agentIndex[idFacility], 'Captial_constructionDuration'] = constructionDuration
    #                rtn.loc[agentIndex[idFacility], 'Captial_Deviation'] = deviation
    #                rtn.loc[agentIndex[idFacility], 'Captial_OvernightCost'] = overnightCost
    #    decommissioning = region.find('decommissioning')
    #    if decommissioning is not None:
    #        duration = int(decommissioning.find('duration').text)
    #        overnightCost = float(decommissioning.find('overnight_cost').text)
    #        rtn.loc[agentIndex[idRegion], 'Decommissioning_Duration'] = duration
    #        rtn.loc[agentIndex[idRegion], 'Decommissioning_OvernightCost'] = overnightCost
    #        for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    #            rtn.loc[agentIndex[idInstitution], 'Decommissioning_Duration'] = duration
    #            rtn.loc[agentIndex[idInstitution], 'Decommissioning_OvernightCost'] = overnightCost
    #            for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    #                rtn.loc[agentIndex[idFacility], 'Decommissioning_Duration'] = duration
    #                rtn.loc[agentIndex[idFacility], 'Decommissioning_OvernightCost'] = overnightCost
    #    operation_maintenance = region.find('operation_maintenance')
    #    if operation_maintenance is not None:
    #        fixed = float(operation_maintenance.find('fixed').text)
    #        variable = float(operation_maintenance.find('variable').text)
    #        deviation = float(operation_maintenance.find('deviation').text)
    #        rtn.loc[agentIndex[idRegion], 'OperationMaintenance_FixedCost'] = fixed
    #        rtn.loc[agentIndex[idRegion], 'OperationMaintenance_VariableCost'] = variable
    #        rtn.loc[agentIndex[idRegion], 'OperationMaintenance_Deviation'] = deviation
    #        for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    #            rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_FixedCost'] = fixed
    #            rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_VariableCost'] = variable
    #            rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_Deviation'] = deviation
    #            for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    #                rtn.loc[agentIndex[idFacility], 'OperationMaintenance_FixedCost'] = fixed
    #                rtn.loc[agentIndex[idFacility], 'OperationMaintenance_VariableCost'] = variable
    #                rtn.loc[agentIndex[idFacility], 'OperationMaintenance_Deviation'] = deviation
    #    fuel = region.find('fuel')
    #    if fuel is not None:
    #        for type in fuel.findall('type'):
    #            supply = float(type.find('supply_cost').text)
    #            waste = float(type.find('waste_fee').text)
    #            name = type.find('name').text
    #            deviation = float(type.find('deviation').text)
    #            if np.isnan(rtn.loc[agentIndex[idRegion], 'Fuel_SupplyCost']):
    #                rtn.loc[agentIndex[idRegion], 'Fuel_Commodity'] = name
    #                rtn.loc[agentIndex[idRegion], 'Fuel_SupplyCost'] = supply
    #                rtn.loc[agentIndex[idRegion], 'Fuel_WasteFee'] = waste
    #                rtn.loc[agentIndex[idRegion], 'Fuel_Deviation'] = deviation
    #            else:
    #                indice = rtn.index.size
    #                rtn.loc[indice] = rtn.loc[agentIndex[idRegion]]
    #                rtn.loc[indice, 'Fuel_Commodity'] = name
    #                rtn.loc[indice, 'Fuel_SupplyCost'] = supply
    #                rtn.loc[indice, 'Fuel_WasteFee'] = waste
    #                rtn.loc[indice, 'Fuel_Deviation'] = deviation
    #        for idInstitution in dfEntry[dfEntry.ParentId==idRegion]['AgentId'].tolist():
    #            if np.isnan(rtn.loc[agentIndex[idInstitution], 'Fuel_SupplyCost']):
    #                rtn.loc[agentIndex[idInstitution], 'Fuel_Commodity'] = name
    #                rtn.loc[agentIndex[idInstitution], 'Fuel_SupplyCost'] = supply
    #                rtn.loc[agentIndex[idInstitution], 'Fuel_WasteFee'] = waste
    #                rtn.loc[agentIndex[idInstitution], 'Fuel_Deviation'] = deviation
    #            else:
    #                indice = rtn.index.size
    #                rtn.loc[indice] = rtn.loc[agentIndex[idInstitution]]
    #                rtn.loc[indice, 'Fuel_Commodity'] = name
    #                rtn.loc[indice, 'Fuel_SupplyCost'] = supply
    #                rtn.loc[indice, 'Fuel_WasteFee'] = waste
    #                rtn.loc[indice, 'Fuel_Deviation'] = deviation
    #        for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    #            if np.isnan(rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost']):
    #                rtn.loc[agentIndex[idFacility], 'Fuel_Commodity'] = name
    #                rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost'] = supply
    #                rtn.loc[agentIndex[idFacility], 'Fuel_WasteFee'] = waste
    #                rtn.loc[agentIndex[idFacility], 'Fuel_Deviation'] = deviation
    #            else:
    #                indice = rtn.index.size
    #                rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    #                rtn.loc[indice, 'Fuel_Commodity'] = name
    #                rtn.loc[indice, 'Fuel_SupplyCost'] = supply
    #                rtn.loc[indice, 'Fuel_WasteFee'] = waste
    #                rtn.loc[indice, 'Fuel_Deviation'] = deviation
    #    for institution in region.findall('institution'):
    #        idInstitution = int(institution.find('id').text)
    #        finance = institution.find('finance')
    #        if finance is not None:
    #            returnOnDebt = float(finance.find('return_on_debt').text)
    #            returnOnEquity = float(finance.find('return_on_equity').text)
    #            taxRate = float(finance.find('tax_rate').text)
    #            discountRate = float(finance.find('discount_rate').text)
    #            rtn.loc[agentIndex[idInstitution], 'Finance_TaxRate'] = taxRate
    #            rtn.loc[agentIndex[idInstitution], 'Finance_ReturnOnDebt'] = returnOnDebt
    #            rtn.loc[agentIndex[idInstitution],'Finance_ReturnOnEquity'] = returnOnEquity
    #            rtn.loc[gent_index[idInstitution], 'Finance_DiscountRate'] = discountRate
    #            for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    #                rtn.loc[agentIndex[idFacility], 'Finance_TaxRate'] = taxRate
    #                rtn.loc[agentIndex[idFacility], 'Finance_ReturnOnDebt'] = returnOnDebt
    #                rtn.loc[agentIndex[idFacility], 'Finance_ReturnOnEquity'] = returnOnEquity
    #                rtn.loc[gent_index[idFacility], 'Finance_DiscountRate'] = discountRate
    #        capital = institution.find('capital')
    #        if capital is not None:
    #            beforePeak = int(capital.find('beforePeak').text)
    #            afterPeak = int(capital.find('afterPeak').text)
    #            constructionDuration = int(capital.find('constructionDuration').text)
    #            deviation = float(capital.find('deviation').text)
    #            overnightCost = float(capital.find('overnight_cost').text)
    #            rtn.loc[agentIndex[idInstitution], 'Captial_beforePeak'] = beforePeak
    #            rtn.loc[agentIndex[idInstitution], 'Captial_afterPeak'] = afterPeak
    #            rtn.loc[agentIndex[idInstitution], 'Captial_constructionDuration'] = constructionDuration
    #            rtn.loc[agentIndex[idInstitution], 'Captial_Deviation'] = deviation
    #            rtn.loc[agentIndex[idInstitution], 'Captial_OvernightCost'] = overnightCost
    #            for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    #                rtn.loc[agentIndex[idFacility], 'Captial_beforePeak'] = beforePeak
    #                rtn.loc[agentIndex[idFacility], 'Captial_constructionDuration'] = constructionDuration
    #                rtn.loc[agentIndex[idFacility], 'Captial_Deviation'] = deviation
    #                rtn.loc[agentIndex[idFacility], 'Captial_OvernightCost'] = overnightCost
    #        decommissioning = institution.find('decommissioning')
    #        if decommissioning is not None:
    #            duration = int(decommissioning.find('duration').text)
    #            overnightCost = float(decommissioning.find('overnight_cost').text)
    #            rtn.loc[agentIndex[idInstitution], 'Decommissioning_Duration'] = duration
    #            rtn.loc[agentIndex[idInstitution], 'Decommissioning_OvernightCost'] = overnightCost
    #            for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    #                rtn.loc[agentIndex[idFacility], 'Decommissioning_Duration'] = duration
    #                rtn.loc[agentIndex[idFacility], 'Decommissioning_OvernightCost'] = overnightCost
    #        operation_maintenance = institution.find('operation_maintenance')
    #        if operation_maintenance is not None:
    #            fixed = float(operation_maintenance.find('fixed').text)
    #            variable = float(operation_maintenance.find('variable').text)
    #            deviation = float(operation_maintenance.find('deviation').text)
    #            rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_FixedCost'] = fixed
    #            rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_VariableCost'] = variable
    #            rtn.loc[agentIndex[idInstitution], 'OperationMaintenance_Deviation'] = deviation
    #            for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    #                rtn.loc[agentIndex[idFacility], 'OperationMaintenance_FixedCost'] = fixed
    #                rtn.loc[agentIndex[idFacility], 'OperationMaintenance_VariableCost'] = variable
    #                rtn.loc[agentIndex[idFacility], 'OperationMaintenance_Deviation'] = deviation
    #        fuel = institution.find('fuel')
    #        if fuel is not None:
    #            for type in fuel.findall('type'):
    #                supply = float(type.find('supply_cost').text)
    #                waste = float(type.find('waste_fee').text)
    #                name = type.find('name').text
    #                deviation = float(type.find('deviation').text)
    #                if np.isnan(rtn.loc[agentIndex[idInstitution], 'Fuel_SupplyCost']):
    #                    rtn.loc[agentIndex[idInstitution], 'Fuel_Commodity'] = name
    #                    rtn.loc[agentIndex[idInstitution], 'Fuel_SupplyCost'] = supply
    #                    rtn.loc[agentIndex[idInstitution], 'Fuel_WasteFee'] = waste
    #                    rtn.loc[agentIndex[idInstitution], 'Fuel_Deviation'] = deviation
    #                else:
    #                    indice = rtn.index.size
    #                    rtn.loc[indice] = rtn.loc[agentIndex[idInstitution]]
    #                    rtn.loc[indice, 'Fuel_Commodity'] = name
    #                    rtn.loc[indice, 'Fuel_SupplyCost'] = supply
    #                    rtn.loc[indice, 'Fuel_WasteFee'] = waste
    #                    rtn.loc[indice, 'Fuel_Deviation'] = deviation
    #                for idFacility in dfEntry[dfEntry.ParentId==idInstitution]['AgentId'].tolist():
    #                    if np.isnan(rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost']):
    #                        rtn.loc[agentIndex[idFacility], 'Fuel_Commodity'] = name
    #                        rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost'] = supply
    #                        rtn.loc[agentIndex[idFacility], 'Fuel_WasteFee'] = waste
    #                        rtn.loc[agentIndex[idFacility], 'Fuel_Deviation'] = deviation
    #                    else:
    #                        indice = rtn.index.size
    #                        rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    #                        rtn.loc[indice, 'Fuel_Commodity'] = name
    #                        rtn.loc[indice, 'Fuel_SupplyCost'] = supply
    #                        rtn.loc[indice, 'Fuel_WasteFee'] = waste
    #                        rtn.loc[indice, 'Fuel_Deviation'] = deviation
    #        for prototype in institution.findall('prototype'):
    #            name = prototype.find('name').text
    #            tmp = dfEntry[dfEntry.ParentId==idInstitution]
    #            facilityIdList = tmp[tmp.Prototype==name].loc[:,'AgentId'].tolist()
    #            capital = prototype.find('capital')
    #            if capital is not None:
    #                beforePeak = int(capital.find('beforePeak').text)
    #                afterPeak = int(capital.find('afterPeak').text)
    #                constructionDuration = int(capital.find('constructionDuration').text)
    #                deviation = float(capital.find('deviation').text)
    #                overnight = float(capital.find('overnight_cost').text)
    #                for idFacility in facilityIdList:
    #                    rtn.loc[agentIndex[idFacility], 'Captial_beforePeak'] = beforePeak
    #                    rtn.loc[agentIndex[idFacility], 'Captial_afterPeak'] = afterPeak
    #                    rtn.loc[agentIndex[idFacility], 'Captial_constructionDuration'] = constructionDuration
    #                    rtn.loc[agentIndex[idFacility], 'Captial_Deviation'] = deviation
    #                    rtn.loc[agentIndex[idFacility], 'Captial_OvernightCost'] = overnight
    #            operation_maintenance = prototype.find('operation_maintenance')
    #            if operation_maintenance is not None:
    #                fixed = float(operation_maintenance.find('fixed').text)
    #                variable = float(operation_maintenance.find('variable').text)
    #                deviation = float(operation_maintenance.find('deviation').text)
    #                for idFacility in facilityIdList:
    #                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_FixedCost'] = fixed
    #                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_VariableCost'] = variable
    #                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_Deviation'] = deviation
    #            fuel = prototype.find('fuel')
    #            if fuel is not None:
    #                for type in fuel.findall('type'):
    #                    supply = float(type.find('supply_cost').text)
    #                    waste = float(type.find('waste_fee').text)
    #                    name = type.find('name').text
    #                    deviation = float(type.find('deviation').text)
    #                    for idFacility in facilityIdList:
    #                        if np.isnan(rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost']):
    #                            rtn.loc[agentIndex[idFacility], 'Fuel_Commodity'] = name
    #                            rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost'] = supply
    #                            rtn.loc[agentIndex[idFacility], 'Fuel_WasteFee'] = waste
    #                            rtn.loc[agentIndex[idFacility], 'Fuel_Deviation'] = deviation
    #                        else:
    #                            indice = rtn.index.size
    #                            rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    #                            rtn.loc[indice, 'Fuel_Commodity'] = name
    #                            rtn.loc[indice, 'Fuel_SupplyCost'] = supply
    #                            rtn.loc[indice, 'Fuel_WasteFee'] = waste
    #                            rtn.loc[indice, 'Fuel_Deviation'] = deviation
    #            decommissioning = prototype.find('decommissioning')
    #            if decommissioning is not None:
    #                duration = int(decommissioning.find('duration').text)
    #                overnight = float(decommissioning.find('overnight_cost').text)
    #                for idFacility in facilityIdList:
    #                    rtn.loc[agentIndex[idFacility], 'Decommissioning_Duration'] = duration
    #                    rtn.loc[agentIndex[idFacility], 'Decommissioning_OvernightCost'] = overnight
    #            for facility in prototype.findall('facility'):
    #                idFacility = int(facility.find('id').text)
    #                capital = facility.find('capital')
    #                if capital is not None:
    #                    rtn.loc[agentIndex[idFacility], 'Captial_beforePeak'] = int(capital.find('beforePeak').text)
    #                    rtn.loc[agentIndex[idFacility], 'Captial_afterPeak'] = int(capital.find('afterPeak').text)
    #                    rtn.loc[agentIndex[idFacility], 'Captial_constructionDuration'] = int(capital.find('constructionDuration').text)
    #                    rtn.loc[agentIndex[idFacility], 'Captial_Deviation'] = float(capital.find('deviation').text)
    #                    rtn.loc[agentIndex[idFacility], 'Captial_OvernightCost'] = float(capital.find('overnight_cost').text)
    #                operation_maintenance = facility.find('operation_maintenance')
    #                if operation_maintenance is not None:
    #                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_FixedCost'] = float(operation_maintenance.find('fixed').text)
    #                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_VariableCost'] = float(operation_maintenance.find('variable').text)
    #                    rtn.loc[agentIndex[idFacility], 'OperationMaintenance_Deviation'] = float(operation_maintenance.find('deviation').text)
    #                fuel = facility.find('fuel')
    #                if fuel is not None:
    #                    for type in fuel.findall('type'):
    #                        supply = float(type.find('supply_cost').text)
    #                        waste = float(type.find('waste_fee').text)
    #                        name = type.find('name').text
    #                        deviation = float(type.find('deviation').text)
    #                        if np.isnan(rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost']):
    #                            rtn.loc[agentIndex[idFacility], 'Fuel_Commodity'] = name
    #                            rtn.loc[agentIndex[idFacility], 'Fuel_SupplyCost'] = supply
    #                            rtn.loc[agentIndex[idFacility], 'Fuel_WasteFee'] = waste
    #                            rtn.loc[agentIndex[idFacility], 'Fuel_Deviation'] = deviation
    #                        else:
    #                            indice = rtn.index.size
    #                            rtn.loc[indice] = rtn.loc[agentIndex[idFacility]]
    #                            rtn.loc[indice, 'Fuel_Commodity'] = name
    #                            rtn.loc[indice, 'Fuel_SupplyCost'] = supply
    #                            rtn.loc[indice, 'Fuel_WasteFee'] = waste
    #                            rtn.loc[indice, 'Fuel_Deviation'] = deviation
    #                decommissioning = facility.find('decommissioning')
    #                if decommissioning is not None:
    #                    rtn.loc[agentIndex[idFacility], 'Decommissioning_Duration'] = int(decommissioning.find('duration').text)
    #                    rtn.loc[agentIndex[idFacility], 'Decommissioning_OvernightCost'] = float(decommissioning.find('overnight_cost').text)
    #return rtn


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
