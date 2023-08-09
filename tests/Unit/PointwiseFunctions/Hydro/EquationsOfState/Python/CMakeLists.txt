# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Hydro.EquationsOfState.Python.Enthalpy"
  "Test_Enthalpy.py"
  "Unit;EquationsOfState;Python"
  PyEquationsOfState)

spectre_add_python_bindings_test(
  "Unit.Hydro.EquationsOfState.Python.PolytropicFluid"
  "Test_PolytropicFluid.py"
  "Unit;EquationsOfState;Python"
  PyEquationsOfState)

spectre_add_python_bindings_test(
  "Unit.Hydro.EquationsOfState.Python.PiecewisePolytropicFluid"
  "Test_PiecewisePolytropicFluid.py"
  "Unit;EquationsOfState;Python"
  PyEquationsOfState)

spectre_add_python_bindings_test(
  "Unit.Hydro.EquationsOfState.Python.Spectral"
  "Test_Spectral.py"
  "Unit;EquationsOfState;Python"
  PyEquationsOfState)
