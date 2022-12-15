# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BoundaryCondition.cpp
  ConstraintPreservingFreeOutflow.cpp
  DirichletAnalytic.cpp
  DirichletFreeOutflow.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCondition.hpp
  ConstraintPreservingFreeOutflow.hpp
  DirichletAnalytic.hpp
  DirichletFreeOutflow.hpp
  Factory.hpp
  ProductOfConditions.hpp
  )
