# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DomainActions")

set(LIBRARY_SOURCES
  Test_InitializeDomain.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/Actions"
  "${LIBRARY_SOURCES}"
  "Domain;Utilities"
  )
