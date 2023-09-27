# Distributed under the MIT License.
# See LICENSE.txt for details.

find_python_module(paraview FALSE)

# Skip tests if ParaView is not found
if (NOT PY_PARAVIEW)
  message(STATUS "ParaView not found, skipping 'Render3D' tests")
  return()
endif()

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.Render3D.Clip"
  Test_Clip.py
  "unit;visualization;python;paraview"
  None)

if (BUILD_PYTHON_BINDINGS)
  set_tests_properties(
    "Unit.Visualization.Python.Render3D.Clip"
    PROPERTIES
    TIMEOUT 30
    )
endif()
