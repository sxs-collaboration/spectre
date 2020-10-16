# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  NewtonianEuler
  PRIVATE
  RegisterDerived.cpp
  Rusanov.cpp
  )

spectre_target_headers(
  NewtonianEuler
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCorrection.hpp
  Factory.hpp
  RegisterDerived.hpp
  Rusanov.hpp
  )
