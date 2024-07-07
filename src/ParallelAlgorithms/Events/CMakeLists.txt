# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Events)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ObserveAdaptiveSteppingDiagnostics.cpp
  ObserveDataBox.cpp
  ObserveNorms.cpp
  ObserveTimeStepVolume.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ErrorIfDataTooBig.hpp
  Factory.hpp
  MonitorMemory.hpp
  ObserveAdaptiveSteppingDiagnostics.hpp
  ObserveDataBox.hpp
  ObserveAtExtremum.hpp
  ObserveFields.hpp
  ObserveNorms.hpp
  ObserveTimeStep.hpp
  ObserveTimeStepVolume.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Charmxx::pup
  LinearOperators
  Observer
  Options
  Parallel
  Serialization
  Time
  Utilities
  INTERFACE
  DataStructures
  Domain
  DomainStructure
  ErrorHandling
  EventsAndTriggers
  H5
  Interpolation
  Printf
  Spectral
  )
