# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory("Python")

set(LIBRARY "Test_Spectral")

set(LIBRARY_SOURCES
  BasisFunctions/Test_Chebyshev.cpp
  BasisFunctions/Test_Fourier.cpp
  BasisFunctions/Test_Jacobi.cpp
  BasisFunctions/Test_Legendre.cpp
  Test_Basis.cpp
  Test_BasisFunctionNormalizationSquare.cpp
  Test_Cartoon.cpp
  Test_ChebyshevGauss.cpp
  Test_ChebyshevGaussLobatto.cpp
  Test_Clenshaw.cpp
  Test_CollocationPointsAndWeights.cpp
  Test_DifferentiationMatrix.cpp
  Test_Filtering.cpp
  Test_FiniteDifference.cpp
  Test_IndefiniteIntegral.cpp
  Test_IntegrationMatrix.cpp
  Test_InterpolationMatrix.cpp
  Test_LegendreGauss.cpp
  Test_LegendreGaussLobatto.cpp
  Test_LinearFilterMatrix.cpp
  Test_LogicalCoordinates.cpp
  Test_Mesh.cpp
  Test_ModalToNodalMatrix.cpp
  Test_NodalToModalMatrix.cpp
  Test_Parity.cpp
  Test_Projection.cpp
  Test_Quadrature.cpp
  Test_QuadratureWeights.cpp
  Test_SegmentSize.cpp
  Test_Spectral.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  DataStructures
  DiscontinuousGalerkin
  Domain
  ErrorHandling
  LinearOperators
  Spectral
  SpectralHelpers
  SphericalHarmonics
  Utilities
  )
