# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.LinearOperators.Python.DefiniteIntegral"
  Test_DefiniteIntegral.py
  "Unit;LinearOperators;Python"
  PyLinearOperators)


spectre_add_python_bindings_test(
  "Unit.LinearOperators.Python.PartialDerivatives"
  Test_PartialDerivatives.py
  "Unit;LinearOperators;Python"
  PyLinearOperators)

spectre_add_python_bindings_test(
  "Unit.LinearOperators.Python.PowerMonitors"
  Test_PowerMonitors.py
  "Unit;LinearOperators;Python"
  PyLinearOperators)
