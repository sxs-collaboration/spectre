# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY XctsSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ConstantDensityStar.cpp
  Flatness.cpp
  Schwarzschild.cpp
  TovStar.cpp
  WrappedGr.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CommonVariables.hpp
  CommonVariables.tpp
  ConstantDensityStar.hpp
  Factory.hpp
  Flatness.hpp
  Schwarzschild.hpp
  TovStar.hpp
  WrappedGr.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ElasticityPointwiseFunctions
  ErrorHandling
  GeneralRelativity
  GeneralRelativitySolutions
  InitialDataUtilities
  Options
  Parallel
  RelativisticEulerSolutions
  Utilities
  Xcts
  XctsAnalyticData
  XctsPointwiseFunctions
  )
