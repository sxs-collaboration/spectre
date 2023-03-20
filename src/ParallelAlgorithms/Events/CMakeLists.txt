# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Events)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  MonitorMemory.hpp
  ObserveAtExtremum.hpp
  ObserveFields.hpp
  ObserveNorms.hpp
  ObserveTimeStep.hpp
  ObserveVolumeIntegrals.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  Domain
  DomainStructure
  ErrorHandling
  EventsAndTriggers
  H5
  Interpolation
  LinearOperators
  Observer
  Options
  Parallel
  Spectral
  Time
  Utilities
  )
