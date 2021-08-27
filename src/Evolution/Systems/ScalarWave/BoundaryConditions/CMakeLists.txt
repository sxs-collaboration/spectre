# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BoundaryCondition.cpp
  ConstraintPreservingSphericalRadiation.cpp
  DirichletAnalytic.cpp
  SphericalRadiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCondition.hpp
  ConstraintPreservingSphericalRadiation.hpp
  DirichletAnalytic.hpp
  Factory.hpp
  SphericalRadiation.hpp
  )
