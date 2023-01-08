# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Domain.Python.BlockAndElementLogicalCoordinates"
  Test_BlockAndElementLogicalCoordinates.py
  "Unit;Domain"
  PyDomain)

spectre_add_python_bindings_test(
  "Unit.Domain.Python.Domain"
  Test_Domain.py
  "Unit;Domain"
  PyDomain)

spectre_add_python_bindings_test(
  "Unit.Domain.Python.ElementId"
  Test_ElementId.py
  "Unit;Domain"
  PyDomain)

spectre_add_python_bindings_test(
  "Unit.Domain.Python.FunctionsOfTime"
  Test_FunctionsOfTime.py
  "Unit;Domain"
  PyDomain)

spectre_add_python_bindings_test(
  "Unit.Domain.Python.InertialCoordinates"
  Test_InertialCoordinates.py
  "Unit;Domain"
  PyDomain)

spectre_add_python_bindings_test(
  "Unit.Domain.Python.SegmentId"
  Test_SegmentId.py
  "Unit;Domain"
  PyDomain)
