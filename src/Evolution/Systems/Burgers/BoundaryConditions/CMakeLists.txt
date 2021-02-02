# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  Burgers
  PRIVATE
  BoundaryCondition.cpp
  DirichletAnalytic.cpp
  Outflow.cpp
  RegisterDerivedWithCharm.cpp
  )

spectre_target_headers(
  Burgers
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCondition.hpp
  DirichletAnalytic.hpp
  Factory.hpp
  Outflow.hpp
  RegisterDerivedWithCharm.hpp
  )
