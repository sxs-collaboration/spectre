# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY XctsAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Binary.cpp
  KerrSchildTeukolsky.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  Binary.hpp
  CommonVariables.hpp
  CommonVariables.tpp
  KerrSchildTeukolsky.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  GeneralRelativity
  GeneralRelativitySolutions
  Hydro
  InitialDataUtilities
  LinearOperators
  Options
  Parallel
  Serialization
  Spectral
  Utilities
  PRIVATE
  ElasticityPointwiseFunctions
  XctsPointwiseFunctions
  )
