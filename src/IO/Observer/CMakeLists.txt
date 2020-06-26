# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  IO
  PRIVATE
  ArrayComponentId.cpp
  ObservationId.cpp
  TypeOfObservation.cpp
  )

spectre_target_headers(
  IO
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ArrayComponentId.hpp
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
