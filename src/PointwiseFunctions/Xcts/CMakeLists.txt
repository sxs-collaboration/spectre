# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY XctsPointwiseFunctions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  LongitudinalOperator.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  LongitudinalOperator.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Utilities
  )
