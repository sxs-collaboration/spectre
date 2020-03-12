# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_FirstOrderDgScheme")

set(LIBRARY_SOURCES
  Test_BoundaryData.cpp
  Test_BoundaryFlux.cpp
  Test_FirstOrderScheme.cpp
  Test_FirstOrderSchemeLts.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder"
  "${LIBRARY_SOURCES}"
  "DataStructures;DiscontinuousGalerkin;Domain;ErrorHandling;Utilities"
  )
