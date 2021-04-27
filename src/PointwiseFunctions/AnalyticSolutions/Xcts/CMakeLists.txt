# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY XctsSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ConstantDensityStar.cpp
  Kerr.cpp
  Schwarzschild.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticSolution.hpp
  CommonVariables.hpp
  CommonVariables.tpp
  ConstantDensityStar.hpp
  Flatness.hpp
  Kerr.hpp
  Schwarzschild.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ElasticityPointwiseFunctions
  ErrorHandling
  GeneralRelativity
  GeneralRelativitySolutions
  Options
  Parallel
  Utilities
  Xcts
  XctsAnalyticData
  XctsPointwiseFunctions
  )
