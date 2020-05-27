# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PoissonDgSchemes")

set(LIBRARY_SOURCES
  Test_FirstOrder.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Poisson/DgSchemes"
  "${LIBRARY_SOURCES}"
  "DataStructures;Poisson;PoissonDgTestHelpers;Utilities"
  )
