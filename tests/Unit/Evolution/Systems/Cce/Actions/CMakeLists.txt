# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Cce_Actions")

set(LIBRARY_SOURCES
  Test_InitializeWorldtubeBoundary.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Cce/Actions/"
  "${LIBRARY_SOURCES}"
  "Cce"
  )
