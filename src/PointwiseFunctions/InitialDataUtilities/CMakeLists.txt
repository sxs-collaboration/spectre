# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY InitialDataUtilities)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  NumericData.cpp
  WithNoise.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticSolution.hpp
  Background.hpp
  InitialData.hpp
  InitialGuess.hpp
  NumericData.hpp
  WithNoise.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  Exporter
  Options
  Serialization
  Spectral
  )

add_subdirectory(Tags)
