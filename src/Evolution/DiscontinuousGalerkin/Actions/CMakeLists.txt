# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApplyBoundaryCorrections.hpp
  BoundaryConditionsImpl.hpp
  ComputeTimeDerivative.hpp
  ComputeTimeDerivativeHelpers.hpp
  InternalMortarDataImpl.hpp
  NormalCovectorAndMagnitude.hpp
  PackageDataImpl.hpp
  VolumeTermsImpl.hpp
  VolumeTermsImpl.tpp
  )
