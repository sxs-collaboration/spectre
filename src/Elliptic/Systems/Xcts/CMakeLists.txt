# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Xcts)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Equations.cpp
  FluxesAndSources.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Equations.hpp
  FirstOrderSystem.hpp
  FluxesAndSources.hpp
  Geometry.hpp
  Tags.hpp
  Xcts.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Utilities
  PRIVATE
  Elasticity
  ErrorHandling
  GeneralRelativity
  LinearOperators
  Poisson
  Utilities
  XctsPointwiseFunctions
  )

add_subdirectory(BoundaryConditions)
