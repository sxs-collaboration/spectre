# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  Evolution
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApplyBoundaryCorrections.hpp
  ComputeTimeDerivative.hpp
  NormalCovectorAndMagnitude.hpp
  )
