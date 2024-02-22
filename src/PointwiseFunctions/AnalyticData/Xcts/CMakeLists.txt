# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY XctsAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Binary.cpp
  BinaryWithGravitationalWaves.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  Binary.hpp
  BinaryWithGravitationalWaves.hpp
  CommonVariables.hpp
  CommonVariables.tpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  GeneralRelativity
  Hydro
  LinearOperators
  Options
  Parallel
  Spectral
  Utilities
  PRIVATE
  ElasticityPointwiseFunctions
  XctsPointwiseFunctions
  )
