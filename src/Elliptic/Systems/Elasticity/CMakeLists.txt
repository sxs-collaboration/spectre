# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Elasticity)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Equations.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Equations.hpp
  FirstOrderSystem.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ConstitutiveRelations
  DataStructures
  Domain
  Utilities
  INTERFACE
  LinearOperators
  )

add_subdirectory(Actions)
add_subdirectory(BoundaryConditions)
