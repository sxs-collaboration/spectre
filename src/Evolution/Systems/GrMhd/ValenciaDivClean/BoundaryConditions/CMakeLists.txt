# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BoundaryCondition.cpp
  DirichletAnalytic.cpp
  Outflow.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCondition.hpp
  DirichletAnalytic.hpp
  Factory.hpp
  Outflow.hpp
  )
