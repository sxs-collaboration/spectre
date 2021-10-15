# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY InitialDataUtilities)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InitialData.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Parallel
  )

add_subdirectory(Tags)
