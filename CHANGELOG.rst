===================
cymetric Change Log
===================

.. current developments

v1.5.3
====================

**Changed:**

* **MAJOR BACKWARDS INCOMPATIBLE CHANGE:** metrics now accept DataFrames, not Series.
  Additionally, the number of arguments in a metric should be equal to the dependencies.
  Dependencies are unpacked into the metric function call.




