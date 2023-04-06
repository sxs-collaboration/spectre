# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Observer)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ArrayComponentId.cpp
  ObservationId.cpp
  ReductionActions.cpp
  TypeOfObservation.cpp
  VolumeActions.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ArrayComponentId.hpp
  GetSectionObservationKey.hpp
  Helpers.hpp
  Initialize.hpp
  ObservationId.hpp
  ObserverComponent.hpp
  ReductionActions.hpp
  Tags.hpp
  TypeOfObservation.hpp
  VolumeActions.hpp
  WriteSimpleData.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  Charmxx::pup
  DataStructures
  Domain
  DomainCreators
  ErrorHandling
  H5
  Options
  Parallel
  Serialization
  Utilities
  INTERFACE
  EventsAndTriggers
  )

add_subdirectory(Actions)
add_subdirectory(Protocols)
