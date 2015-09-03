Performing economic analysis with the economic tools provided by Cymetric
=========================================================================


These are the main python files to perform an economic analysis using Cymetric :

| - eco_metrics.py
| - eco_tools.py
| - test_eco_metrics.py

Economic metrics calculation
----------------------------
eco_metrics.py has the same goal as metrics.py (ie it contains metrics), but it is only containing the economic metrics.
These metrics correspond to the reactor costs (construction costs, O&M costs, fuel costs and decommissioning costs).
Other functions allow the user to do more complex calculations (annual costs, LCOE...) and are written in eco_metrics.py as well. However, they are not actual metrics, because they do not have the @metric decorator. These are only functions whose results will not be stored in the output database (unlike the 'real' metrics with decorator).

Additional tools needed for analysis
------------------------------------
In order to support the calculations written in eco_metrics.py, some functions are stored in eco_tools.py.

Tests
-----
Tests are written in test_eco_metrics.py. The tests are inspired by test_metrics.py.

Using these functions
---------------------

In order to perform economic analysis, you can follow these steps :

1) Run your simulation with Cyclus

2) Write an xml file with the economic data needed to calculate the economic metrics and save it as 'parameters.xml' (see eco_parameters_frame.xml) in the same folder as the sqlite output database.

3) Calculate the different metrics you are interested in :

| - EconomicInfo to store the economic parameters needed for the economic analysis
| - CapitalCost to calculate the cash flows corresponding to the construction costs
| - OperationMaintenance to calculate the cash flows corresponding to the operations and maintenance costs
| - FuelCost to calculate the cash flows corresponding to the fuel costs
| - DecommissioningCost to calculate the costs corresponding to decommissionings occurring during the simulation

