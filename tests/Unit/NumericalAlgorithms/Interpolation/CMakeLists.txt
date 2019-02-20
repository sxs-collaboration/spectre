# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NumericalInterpolation")

set(LIBRARY_SOURCES
  Test_AddTemporalIdsToInterpolationTarget.cpp
  Test_BarycentricRational.cpp
  Test_CleanUpInterpolator.cpp
  Test_InitializeInterpolationTarget.cpp
  Test_InitializeInterpolator.cpp
  Test_InterpolationTargetApparentHorizon.cpp
  Test_InterpolationTargetKerrHorizon.cpp
  Test_InterpolationTargetLineSegment.cpp
  Test_InterpolationTargetReceiveVars.cpp
  Test_InterpolationTargetWedgeSectionTorus.cpp
  Test_InterpolatorReceivePoints.cpp
  Test_InterpolatorReceiveVolumeData.cpp
  Test_InterpolatorRegisterElement.cpp
  Test_IrregularInterpolant.cpp
  Test_LagrangePolynomial.cpp
  Test_ObserveTimeSeriesOnSurface.cpp
  Test_ParallelInterpolator.cpp
  Test_RegularGridInterpolant.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/Interpolation/"
  "${LIBRARY_SOURCES}"
  "Interpolation;MathFunctions"
  )
