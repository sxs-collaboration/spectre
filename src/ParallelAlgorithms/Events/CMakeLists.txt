# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Events)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Completion.cpp
  ObserveAdaptiveSteppingDiagnostics.cpp
  ObserveConstantsPerElement.cpp
  ObserveDataBox.cpp
  ObserveGridStats.cpp
  ObserveNorms.cpp
  ObserveTimeStep.cpp
  ReductionScope.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ChangeFixedLtsRatio.hpp
  Completion.hpp
  ErrorIfDataTooBig.hpp
  Factory.hpp
  MonitorMemory.hpp
  ObserveAdaptiveSteppingDiagnostics.hpp
  ObserveConstantsPerElement.hpp
  ObserveDataBox.hpp
  ObserveAtExtremum.hpp
  ObserveFields.hpp
  ObserveGridStats.hpp
  ObserveNorms.hpp
  ObserveTimeStep.hpp
  ObserveTimeStep.tpp
  ObserveTimeStepVolume.hpp
  ObserveTimeStepVolume.tpp
  ReductionScope.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Charmxx::pup
  DomainStructure
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
  DomainCreators
  ErrorHandling
  EventsAndTriggers
  H5
  Interpolation
  Printf
  Spectral
  )
