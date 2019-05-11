# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Spectral")

set(LIBRARY_SOURCES
  SwshTestHelpers.cpp
  Test_ChebyshevGauss.cpp
  Test_ChebyshevGaussLobatto.cpp
  Test_ComplexDataView.cpp
  Test_Filtering.cpp
  Test_IndefiniteIntegral.cpp
  Test_LegendreGauss.cpp
  Test_LegendreGaussLobatto.cpp
  Test_Projection.cpp
  Test_Spectral.cpp
  Test_SwshCoefficients.cpp
  Test_SwshCollocation.cpp
  Test_SwshTags.cpp
  Test_SwshTestHelpers.cpp
  Test_SwshTransformJob.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/Spectral/"
  "${LIBRARY_SOURCES}"
  "Spectral;Domain"
  )
