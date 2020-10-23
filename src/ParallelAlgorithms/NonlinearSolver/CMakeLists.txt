# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelNonlinearSolver)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ErrorHandling
  Utilities
  INTERFACE
  DataStructures
  Options
  )

add_subdirectory(NewtonRaphson)
