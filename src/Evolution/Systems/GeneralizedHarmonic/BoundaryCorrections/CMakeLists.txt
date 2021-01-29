# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  GeneralizedHarmonic
  PRIVATE
  RegisterDerived.cpp
  UpwindPenalty.cpp
  )

spectre_target_headers(
  GeneralizedHarmonic
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCorrection.hpp
  Factory.hpp
  RegisterDerived.hpp
  UpwindPenalty.hpp
  )
