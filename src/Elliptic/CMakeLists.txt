# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Elliptic)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ErrorHandling
  Options
  INTERFACE
  DataStructures
  DiscontinuousGalerkin
  Domain
  EventsAndTriggers
  LinearOperators
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(BoundaryConditions)
add_subdirectory(DiscontinuousGalerkin)
add_subdirectory(Executables)
add_subdirectory(Systems)
add_subdirectory(Triggers)
add_subdirectory(Utilities)
