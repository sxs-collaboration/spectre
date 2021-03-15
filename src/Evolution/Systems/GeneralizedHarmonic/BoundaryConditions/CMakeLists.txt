# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  GeneralizedHarmonic
  PRIVATE
  BjorhusImpl.cpp
  BoundaryCondition.cpp
  DirichletAnalytic.cpp
  RegisterDerivedWithCharm.cpp
  )

spectre_target_headers(
  GeneralizedHarmonic
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BjorhusImpl.hpp
  BoundaryCondition.hpp
  DirichletAnalytic.hpp
  Factory.hpp
  RegisterDerivedWithCharm.hpp
  )
