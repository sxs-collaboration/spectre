# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NumericalDiscontinuousGalerkinTags")

set(LIBRARY_SOURCES
  Test_Formulation.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/DiscontinuousGalerkin/Tags/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;DiscontinuousGalerkin;ErrorHandling;Spectral;Utilities"
  )
