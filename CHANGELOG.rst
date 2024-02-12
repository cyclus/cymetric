===================
cymetric Change Log
===================

.. current developments
**Added:**
* GitHub workflows for CI (#188, #190, #191, #193)

**Changed**
* Converted test suite from nose to pytest (#188)

v1.5.5
====================

**Added:**

* Monthly electricity generated metric. One for returning electricity by one agent and another for all agents.
* Added units parameter.


v1.5.4
====================

**Added:**
* Capability to have either the commodity name and/or the mass quantity
  transferred between 2 facilities in the flow graph arrows

**Changed:**

* Switching CI to circle-CI 2.0
* Improved PandaDataframe filtering


**Fixed:**

* Improper input file for the nosetest


v1.5.3
====================

**Changed:**

* **MAJOR BACKWARDS INCOMPATIBLE CHANGE:** metrics now accept DataFrames, not Series.
  Additionally, the number of arguments in a metric should be equal to the dependencies.
  Dependencies are unpacked into the metric function call.




