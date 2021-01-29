# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ScalarWave
  PRIVATE
  RegisterDerived.cpp
  UpwindPenalty.cpp
  )

spectre_target_headers(
  ScalarWave
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCorrection.hpp
  Factory.hpp
  RegisterDerived.hpp
  UpwindPenalty.hpp
  )
