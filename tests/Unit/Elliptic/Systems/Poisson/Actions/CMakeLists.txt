# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PoissonActions")

set(LIBRARY_SOURCES
  Test_Observe.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Poisson/Actions/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;ErrorHandling"
  )
