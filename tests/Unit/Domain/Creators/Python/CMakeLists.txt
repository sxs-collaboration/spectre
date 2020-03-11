# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.Python.Brick"
  Test_Brick.py
  "Unit;Domain")

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.Python.Cylinder"
  Test_Cylinder.py
  "Unit;Domain")

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.Python.Interval"
  Test_Interval.py
  "Unit;Domain")

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.Python.Rectangle"
  Test_Rectangle.py
  "Unit;Domain")
