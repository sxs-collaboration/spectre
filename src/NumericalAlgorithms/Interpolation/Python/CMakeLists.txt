# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyInterpolation")

spectre_python_add_module(
  Interpolation
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  BarycentricRational.cpp
  Bindings.cpp
  CubicSpline.cpp
  IrregularInterpolant.cpp
  RegularGridInterpolant.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BarycentricRational.hpp
  CubicSpline.hpp
  IrregularInterpolant.hpp
  RegularGridInterpolant.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  Interpolation
  DataStructures
  Spectral
  pybind11::module
)

spectre_python_add_dependencies(${LIBRARY} PyDataStructures PySpectral)
