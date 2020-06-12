# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Domain)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Block.cpp
  BlockLogicalCoordinates.cpp
  CreateInitialElement.cpp
  Domain.cpp
  DomainHelpers.cpp
  ElementLogicalCoordinates.cpp
  ElementMap.cpp
  FaceNormal.cpp
  LogicalCoordinates.cpp
  MinimumGridSpacing.cpp
  SizeOfElement.cpp
  SurfaceJacobian.cpp
  Tags.cpp
  TagsTimeDependent.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Block.hpp
  BlockLogicalCoordinates.hpp
  CreateInitialElement.hpp
  Domain.hpp
  DomainHelpers.hpp
  ElementLogicalCoordinates.hpp
  ElementMap.hpp
  FaceNormal.hpp
  InterfaceComputeTags.hpp
  InterfaceHelpers.hpp
  LogicalCoordinates.hpp
  MinimumGridSpacing.hpp
  OptionTags.hpp
  SizeOfElement.hpp
  Tags.hpp
  TagsCharacteresticSpeeds.hpp
  TagsTimeDependent.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  DomainStructure
  ErrorHandling
  Options
  Utilities
  INTERFACE
  FunctionsOfTime
  Time
  CoordinateMaps
  DomainCreators
  Spectral
  )

add_subdirectory(Amr)
add_subdirectory(CoordinateMaps)
add_subdirectory(Creators)
add_subdirectory(FunctionsOfTime)
add_subdirectory(Python)
add_subdirectory(Structure)
