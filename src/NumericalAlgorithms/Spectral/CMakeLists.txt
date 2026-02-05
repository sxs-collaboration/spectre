# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Spectral)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BarycentricWeights.cpp
  Basis.cpp
  BoundaryInterpolationMatrices.cpp
  BoundaryInterpolationTerm.cpp
  BoundaryLiftingTerm.cpp
  Cartoon.cpp
  CollocationPoints.cpp
  CollocationPointsAndWeights.cpp
  DifferentiationMatrix.cpp
  Filtering.cpp
  FiniteDifference.cpp
  IntegrationMatrix.cpp
  InterpolationMatrix.cpp
  LinearFilteringMatrix.cpp
  LogicalCoordinates.cpp
  Mesh.cpp
  ModalToNodalMatrix.cpp
  NodalToModalMatrix.cpp
  Parity.cpp
  Projection.cpp
  Quadrature.cpp
  QuadratureWeights.cpp
  SegmentSize.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BarycentricWeights.hpp
  Basis.hpp
  BasisFunctionNormalizationSquare.hpp
  BasisFunctionValue.hpp
  BoundaryInterpolationMatrices.hpp
  BoundaryInterpolationTerm.hpp
  BoundaryLiftingTerm.hpp
  Clenshaw.hpp
  CollocationPoints.hpp
  CollocationPointsAndWeights.hpp
  DifferentiationMatrix.hpp
  Filtering.hpp
  GetSpectralQuantityForMesh.hpp
  IntegrationMatrix.hpp
  InterpolationMatrix.hpp
  InverseWeightFunctionValues.hpp
  LinearFilteringMatrix.hpp
  LogicalCoordinates.hpp
  MaximumNumberOfPoints.hpp
  Mesh.hpp
  MinimumNumberOfPoints.hpp
  ModalToNodalMatrix.hpp
  Namespace.hpp
  NodalToModalMatrix.hpp
  Parity.hpp
  PrecomputedSpectralQuantity.hpp
  Projection.hpp
  Quadrature.hpp
  QuadratureWeights.hpp
  SegmentSize.hpp
  SpectralQuantityForMesh.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  Utilities
  PRIVATE
  BLAS::BLAS
  LAPACK::LAPACK
  RootFinding
  SPHEREPACK
  )

add_subdirectory(BasisFunctions)
add_subdirectory(Python)
