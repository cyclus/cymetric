Performing economic analysis with the economic tools provided by Cymetric
=========================================================================


Several functions are especially dedicated to the economic analysis of fuel cycles using Cymetric. The related files are the following :

| - eco_metrics.py
| - eco_inputs.py
| - cash_flows.py
| - test_eco_metrics.py

Economic metrics calculation
----------------------------
eco_metrics.py has the same goal as metrics.py (ie it contains metrics), only it is only containing the economic metrics.
Until now, these metrics are all related to reactor costs (capital cost, fixed O&M cost, decommissioning cost, variable O&M cost, fuel cost and waste fee). As for the front end and the back end of the fuel cycle, prices will be fixed at realistic values according to actual values. This is because prices should be  calculated by the Dynamic Resource Exchange.

Additional tools needed for analysis
------------------------------------
In order to make the analysis more complex and realistic, some tools are stored in eco_inputs.py.
First, some financial parameters are used in order to calculate realistic costs. These parameters are then imported in eco_metrics.py.
Second, eco_inputs.py contains a few functions that mainly add complexity to the calculation of the metrics in eco_metrics.py.

Cash flows visualization
------------------------
Given all the fuel cycle costs, we are able to gather them all to calculate the monthly/annual total cash flows. We could also calculate important indicators such as the levelized cost of electricity. This features are stored in cash_flows/lcoe.py. 

Tests
-----
Tests are stored in test_eco_metrics.py. The tests are inspired by test_metrics.py.

Using these functions
---------------------

In order to perform economic analysis, you can follow these steps :

1) Run your simulation with Cyclus

2) Fill an xml file with the economic data needed to calculate the economic metrics and save it as 'parameters.xml' in the folder as your workspace (probably where the output database of the simulation is) where you will investigate the output data.

3) Calculate the different metrics you are interested in :

| - EconomicInfo to see what economic parameters are used for the analysis
| - CapitalCost to see the cash flows related to the construction costs
| - OperationMaintenance to see the cash flows related to the operations and maintenance
| - FuelCost to see cash flows related to the fuel costs
| - DecommissioningCost to see the costs related to decommissionings occurring during the simulation

4) Use the functions of cash_flows.py to calculate more complex metrics (at agent, institution, region or simulation level) :

| - Annual costs
| - Levelized Cost Of Electricity
| - Average Levelized Cost Of Electricity
| - Accumulated capital
| - Period Costs

5) There are also some functions dedicated to plot these metrics in cash_flows.py

