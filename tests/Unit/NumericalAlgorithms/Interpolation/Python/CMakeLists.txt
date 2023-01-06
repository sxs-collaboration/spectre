# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Interpolation.Python.BarycentricRational"
  Test_BarycentricRational.py
  "Unit;Interpolation;Python"
  PyInterpolation)

spectre_add_python_bindings_test(
  "Unit.Interpolation.Python.CubicSpline"
  Test_CubicSpline.py
  "Unit;Interpolation;Python"
  PyInterpolation)

spectre_add_python_bindings_test(
  "Unit.Interpolation.Python.RegularGridInterpolant"
  Test_RegularGridInterpolant.py
  "Unit;Interpolation;Python"
  PyInterpolation)

spectre_add_python_bindings_test(
  "Unit.Interpolation.Python.IrregularInterpolant"
  Test_IrregularInterpolant.py
  "Unit;Interpolation;Python"
  PyInterpolation)
