# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Events)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  ObserveErrorNorms.hpp
  ObserveFields.hpp
  ObserveNorms.hpp
  ObserveTimeStep.hpp
  ObserveVolumeIntegrals.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  Domain
  DomainStructure
  ErrorHandling
  EventsAndTriggers
  Interpolation
  LinearOperators
  Options
  Time
  Utilities
  )
