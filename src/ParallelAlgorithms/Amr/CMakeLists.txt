# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelAmr)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Amr
  AmrCriteria
  AmrPolicies
  AmrProjectors
  DataStructures
  Initialization
  Logging
  Printf
  Serialization
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(Criteria)
add_subdirectory(Policies)
add_subdirectory(Projectors)
add_subdirectory(Protocols)
