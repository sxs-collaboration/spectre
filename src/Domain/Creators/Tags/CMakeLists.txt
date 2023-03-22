# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Domain.cpp
  InitialExtents.cpp
  InitialRefinementLevels.cpp
  ObjectCenter.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Domain.hpp
  ExternalBoundaryConditions.hpp
  FunctionsOfTime.hpp
  InitialExtents.hpp
  InitialRefinementLevels.hpp
  ObjectCenter.hpp
  )
