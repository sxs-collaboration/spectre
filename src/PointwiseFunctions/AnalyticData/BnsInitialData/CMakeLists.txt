# Distributed under the MIT License.
# See LICENSE.txt for details.
set(LIBRARY BnsInitialDataAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  SpectreData.cpp
  )
spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  SpectreData.hpp
  Factory.hpp
  )
target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  BnsInitialData
  ErrorHandling
  Exporter
  GeneralRelativity
  GeneralRelativityAnalyticData
  Options
  Utilities
  LinearOperators
)

