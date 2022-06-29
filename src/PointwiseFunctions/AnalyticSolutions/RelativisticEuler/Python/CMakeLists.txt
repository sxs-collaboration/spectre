# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyRelativisticEulerSolutions")

spectre_python_add_module(
  RelativisticEuler
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "PointwiseFunctions/AnalyticSolutions"
  SOURCES
  Bindings.cpp
  Tov.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  RelativisticEulerSolutions
  Hydro
  pybind11::module
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyEquationsOfState
  )
