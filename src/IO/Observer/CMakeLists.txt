# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ArrayComponentId.cpp
  ObservationId.cpp
  ReductionActions.cpp
  TypeOfObservation.cpp
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

add_subdirectory(Actions)
add_subdirectory(Protocols)
