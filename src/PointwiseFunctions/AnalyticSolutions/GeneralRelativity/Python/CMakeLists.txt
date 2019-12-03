# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyGeneralRelativitySolutions")

spectre_python_add_module(
  GeneralRelativity
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "PointwiseFunctions/AnalyticSolutions"
  SOURCES
  Bindings.cpp
  Tov.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PUBLIC GeneralRelativitySolutions
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyEquationsOfState
  )
