**Added:** None

**Changed:**

* **MAJOR BACKWARDS INCOMPATIBLE CHANGE:** metrics now accept DataFrames, not Series.
  Additionally, the number of arguments in a metric should be equal to the dependencies.
  Dependencies are unpacked into the metric function call.

**Deprecated:** None

**Removed:** None

**Fixed:** None

**Security:** None
