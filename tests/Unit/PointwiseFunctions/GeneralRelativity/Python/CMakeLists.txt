# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.GeneralRelativity.Python"
  Test_Bindings.py
  "Unit;GeneralRelativity;Python"
  PyGeneralRelativity)

spectre_add_python_bindings_test(
  "Unit.GeneralizedHarmonic.Python"
  Test_GhBindings.py
  "Unit;GeneralizedHarmonic;Python"
  PyGeneralizedHarmonic)
