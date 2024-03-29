# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_LinearOperators")

set(LIBRARY_SOURCES
  Test_CoefficientTransforms.cpp
  Test_DefiniteIntegral.cpp
  Test_Divergence.cpp
  Test_Filtering.cpp
  Test_IndefiniteIntegral.cpp
  Test_Linearize.cpp
  Test_MeanValue.cpp
  Test_PartialDerivatives.cpp
  Test_PowerMonitors.cpp
  Test_WeakDivergence.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Actions
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainCreators
  ErrorHandling
  LinearOperators
  MathFunctions
  Spectral
  Utilities
  )

add_subdirectory(Python)
