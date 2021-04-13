# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BoundaryConditionType.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticSolution.hpp
  ApplyBoundaryCondition.hpp
  BoundaryCondition.hpp
  BoundaryConditionType.hpp
  Tags.hpp
  )
