# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  NewtonianEuler
  PRIVATE
  BoundaryCondition.cpp
  DirichletAnalytic.cpp
  RegisterDerivedWithCharm.cpp
  )

spectre_target_headers(
  NewtonianEuler
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCondition.hpp
  DirichletAnalytic.hpp
  Factory.hpp
  RegisterDerivedWithCharm.hpp
  )
