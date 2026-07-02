# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PySpinWeightedSphericalHarmonics")

spectre_python_add_module(
  SpinWeightedSphericalHarmonics
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  SwshCoefficients.cpp
  SwshDerivatives.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InterleavedHelpers.hpp
  SwshCoefficients.hpp
  SwshDerivatives.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  SpinWeightedSphericalHarmonics
  pybind11::module
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  )
