# Distributed under the MIT License.
# See LICENSE.txt for details.

option(ENABLE_PARAVIEW "Try to find ParaView to enable 3D rendering tools" OFF)

if (NOT ENABLE_PARAVIEW)
  return()
endif()

find_package(ParaView REQUIRED)

# Help `find_python_module` find ParaView
if (PARAVIEW_PYTHONPATH)
  set(PY_paraview_LOCATION ${PARAVIEW_PYTHONPATH}/paraview)
endif()
