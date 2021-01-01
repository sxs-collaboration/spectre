# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  Elliptic
  PRIVATE
  BoundaryConditionType.cpp
  )

spectre_target_headers(
  Elliptic
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApplyBoundaryCondition.hpp
  BoundaryCondition.hpp
  BoundaryConditionType.hpp
  Tags.hpp
  )
