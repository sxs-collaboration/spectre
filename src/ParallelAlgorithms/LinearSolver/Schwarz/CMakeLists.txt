# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelSchwarz)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  OverlapHelpers.cpp
  Weighting.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ElementCenteredSubdomainData.hpp
  OverlapHelpers.hpp
  Schwarz.hpp
  Weighting.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  DomainStructure
  Utilities
  PRIVATE
  ErrorHandling
  INTERFACE
  LinearSolver
  )

add_subdirectory(SubdomainPreconditioners)
