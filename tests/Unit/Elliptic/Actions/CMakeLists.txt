# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticActions")

set(LIBRARY_SOURCES
  Test_ComputeOperatorAction.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Actions/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;ErrorHandling"
  )
