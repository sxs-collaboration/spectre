# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.GenerateXdmf"
  Test_GenerateXdmf.py
  "unit;visualization;python"
  None)

spectre_add_python_test(
  "Unit.Visualization.Python.Render1D"
  Test_Render1D.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.InterpolateVolumeData"
  Test_InterpolateVolumeData.py
  "unit;visualization;python"
  None)
