# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_FiniteDifference")

set(LIBRARY_SOURCES
  Test_AoWeno53.cpp
  Test_DerivativeOrder.cpp
  Test_FallbackReconstructorType.cpp
  Test_Filter.cpp
  Test_HighOrderFluxCorrection.cpp
  Test_Minmod.cpp
  Test_MonotonicityPreserving5.cpp
  Test_MonotonisedCentral.cpp
  Test_NeighborDataAsVariables.cpp
  Test_NonUniform1D.cpp
  Test_PartialDerivatives.cpp
  Test_PositivityPreservingAdaptiveOrder.cpp
  Test_Unlimited.cpp
  Test_Wcns5z.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/FiniteDifference/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  DgSubcell
  Domain
  FiniteDifference
  ErrorHandling
  Utilities
  )
