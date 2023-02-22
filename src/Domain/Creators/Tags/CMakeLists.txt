# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PUBLIC
  Domain.cpp
  InitialExtents.cpp
  InitialRefinementLevels.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BlockNamesAndGroups.hpp
  Domain.hpp
  InitialExtents.hpp
  InitialRefinementLevels.hpp
  )
