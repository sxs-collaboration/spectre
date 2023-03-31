# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.GenerateXdmf"
  Test_GenerateXdmf.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotDatFile"
  Test_PlotDatFile.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotPowerMonitors"
  Test_PlotPowerMonitors.py
  "unit;visualization;python"
  None)

if(${BUILD_PYTHON_BINDINGS})
  # Test is a bit slow because it writes a bunch of plot files to verify
  # the argument handling works.
  set_tests_properties(
    "Unit.Visualization.Python.PlotDatFile"
    PROPERTIES
    TIMEOUT 10
    )
  set_tests_properties(
    "Unit.Visualization.Python.PlotPowerMonitors"
    PROPERTIES
    TIMEOUT 10
    )
endif()

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.ReadH5"
  Test_ReadH5.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.ReadInputFile"
  Test_ReadInputFile.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.Render1D"
  Test_Render1D.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.InterpolateToCoords"
  Test_InterpolateToCoords.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.InterpolateToMesh"
  Test_InterpolateToMesh.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.TransformVolumeData"
  Test_TransformVolumeData.py
  "unit;visualization;python"
  None)
