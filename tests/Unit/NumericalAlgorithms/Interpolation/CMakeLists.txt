# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Python)

set(LIBRARY "Test_NumericalInterpolation")

set(LIBRARY_SOURCES
  Test_BarycentricRational.cpp
  Test_CardinalInterpolator.cpp
  Test_CubicSpline.cpp
  Test_IrregularInterpolant.cpp
  Test_LagrangePolynomial.cpp
  Test_MultiLinearSpanInterpolation.cpp
  Test_LinearRegression.cpp
  Test_PolynomialInterpolation.cpp
  Test_PredictedZeroCrossing.cpp
  Test_RegularGridInterpolant.cpp
  Test_SendGhWorldtubeData.cpp
  Test_SpanInterpolators.cpp
  Test_ZeroCrossingPredictor.cpp
  )

if(Boost_VERSION VERSION_GREATER_EQUAL 1.81.0)
  list(APPEND LIBRARY_SOURCES Test_UniformCardinalBSpline.cpp)
endif()

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  CoordinateMaps
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  GeneralRelativitySolutions
  Interpolation
  LinearAlgebra
  Logging
  MathFunctions
  ParallelInterpolation
  RelativisticEulerSolutions
  Spectral
  SpectralHelpers
  SphericalHarmonicsHelpers
  SpinWeightedSphericalHarmonics
  Utilities
  )
