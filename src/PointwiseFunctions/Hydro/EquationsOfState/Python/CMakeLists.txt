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
  Enthalpy.cpp
  PiecewisePolytropicFluid.cpp
  PolytropicFluid.cpp
  Spectral.cpp
  PYTHON_FILES
  __init__.py
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  EquationOfState.hpp
  Enthalpy.hpp
  PiecewisePolytropicFluid.hpp
  PolytropicFluid.hpp
  Spectral.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Hydro
  pybind11::module
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
  )
