# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Python)

set(LIBRARY "Test_NumericalInterpolation")

set(LIBRARY_SOURCES
  Test_BarycentricRational.cpp
  Test_CubicSpline.cpp
  Test_IrregularInterpolant.cpp
  Test_LagrangePolynomial.cpp
  Test_RegularGridInterpolant.cpp
  Test_SendGhWorldtubeData.cpp
  Test_SpanInterpolators.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/Interpolation/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  CoordinateMaps
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  GeneralRelativitySolutions
  Interpolation
  IO
  Logging
  MathFunctions
  RelativisticEulerSolutions
  Spectral
  Utilities
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
