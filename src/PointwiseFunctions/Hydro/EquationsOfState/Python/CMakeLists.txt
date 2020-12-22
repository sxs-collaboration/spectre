# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyEquationsOfState")

spectre_python_add_module(
  EquationsOfState
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "PointwiseFunctions/Hydro"
  SOURCES
  Bindings.cpp
  EquationOfState.cpp
  PolytropicFluid.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Hydro
  pybind11::module
  )
