# Distributed under the MIT License.
# See LICENSE.txt for details.

option(ENABLE_PARAVIEW "Try to find ParaView to enable 3D rendering tools" ON)

if (ENABLE_PARAVIEW)
  find_package(ParaView)
endif()

# Help `find_python_module` find ParaView
if (PARAVIEW_PYTHONPATH)
  set(PY_paraview_LOCATION ${PARAVIEW_PYTHONPATH}/paraview)
endif()
