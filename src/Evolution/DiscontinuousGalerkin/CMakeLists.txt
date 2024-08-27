# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AtomicInboxBoundaryData.hpp
  BackgroundGrVars.hpp
  BoundaryData.hpp
  DgElementArray.hpp
  InboxTags.hpp
  MortarData.hpp
  MortarDataHolder.hpp
  MortarTags.hpp
  NormalVectorTags.hpp
  UsingSubcell.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AtomicInboxBoundaryData.cpp
  BoundaryData.cpp
  MortarData.cpp
  MortarDataHolder.cpp
  )

add_subdirectory(Actions)
add_subdirectory(Initialization)
add_subdirectory(Limiters)
add_subdirectory(Messages)
