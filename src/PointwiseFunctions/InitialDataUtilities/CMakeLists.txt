# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY InitialDataUtilities)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticSolution.hpp
  Background.hpp
  InitialData.hpp
  InitialGuess.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Serialization
  )

add_subdirectory(Tags)
