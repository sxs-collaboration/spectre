# Distributed under the MIT License.
# See LICENSE.txt for details.

find_python_module(paraview)

# Skip tests if ParaView is not found
if (NOT PY_paraview_FOUND)
  message(STATUS "ParaView not found, skipping 'Render3D' tests")
  return()
endif()

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.Render3D.Clip"
  Test_Clip.py
  "unit;visualization;python;paraview"
  None
  TIMEOUT 30)

spectre_add_python_bindings_test(
  "Unit.Visualization.Python.Render3D.Domain"
  Test_Domain.py
  "unit;visualization;python;paraview"
  None
  TIMEOUT 30)
