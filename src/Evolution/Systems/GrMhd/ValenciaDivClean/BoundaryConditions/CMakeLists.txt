# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ValenciaDivClean
  PRIVATE
  BoundaryCondition.cpp
  DirichletAnalytic.cpp
  RegisterDerivedWithCharm.cpp
  )

spectre_target_headers(
  ValenciaDivClean
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCondition.hpp
  DirichletAnalytic.hpp
  Factory.hpp
  RegisterDerivedWithCharm.hpp
  )
