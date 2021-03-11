# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ScalarWave
  PRIVATE
  BoundaryCondition.cpp
  ConstraintPreservingSphericalRadiation.cpp
  DirichletAnalytic.cpp
  RegisterDerivedWithCharm.cpp
  SphericalRadiation.cpp
  )

spectre_target_headers(
  ScalarWave
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCondition.hpp
  ConstraintPreservingSphericalRadiation.hpp
  DirichletAnalytic.hpp
  Factory.hpp
  RegisterDerivedWithCharm.hpp
  SphericalRadiation.hpp
  )
