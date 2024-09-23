# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.GenerateTetrahedralConnectivity"
  Test_GenerateTetrahedralConnectivity.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.GenerateXdmf"
  Test_GenerateXdmf.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotAlongLine"
  Test_PlotAlongLine.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotCce"
  Test_PlotCce.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotControlSystem"
  Test_PlotControlSystem.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotDatFile"
  Test_PlotDatFile.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotEllipticConvergence"
  Test_PlotEllipticConvergence.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotMemoryMonitors"
  Test_PlotMemoryMonitors.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotPowerMonitors"
  Test_PlotPowerMonitors.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotSizeControl"
  Test_PlotSizeControl.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotSlice"
  Test_PlotSlice.py
  "unit;visualization;python"
  None)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.PlotTrajectories"
  Test_PlotTrajectories.py
  "unit;visualization;python"
  None)

if(${BUILD_PYTHON_BINDINGS})
  # Test is a bit slow because it writes a bunch of plot files to verify
  # the argument handling works.
  set_tests_properties(
    "Unit.Visualization.Python.PlotAlongLine"
    PROPERTIES
    TIMEOUT 10
    )
  set_tests_properties(
    "Unit.Visualization.Python.PlotCce"
    PROPERTIES
    TIMEOUT 10
    )
  set_tests_properties(
    "Unit.Visualization.Python.PlotControlSystem"
    PROPERTIES
    TIMEOUT 10
    )
  set_tests_properties(
    "Unit.Visualization.Python.PlotDatFile"
    PROPERTIES
    TIMEOUT 10
    )
  set_tests_properties(
    "Unit.Visualization.Python.PlotEllipticConvergence"
    PROPERTIES
    TIMEOUT 10
    )
  set_tests_properties(
    "Unit.Visualization.Python.PlotMemoryMonitors"
    PROPERTIES
    TIMEOUT 10
    )
  set_tests_properties(
    "Unit.Visualization.Python.PlotPowerMonitors"
    PROPERTIES
    TIMEOUT 10
    )
  set_tests_properties(
    "Unit.Visualization.Python.PlotSizeControl"
    PROPERTIES
    TIMEOUT 10
    )
  set_tests_properties(
    "Unit.Visualization.Python.PlotSlice"
    PROPERTIES
    TIMEOUT 10
    )
  set_tests_properties(
    "Unit.Visualization.Python.PlotTrajectories"
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

if(${BUILD_PYTHON_BINDINGS})
  set_tests_properties(
    "Unit.Visualization.Python.Render1D"
    PROPERTIES
    TIMEOUT 10
  )
endif()

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

add_subdirectory(Render3D)
