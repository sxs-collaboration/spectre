# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DomainStructure")

set(LIBRARY_SOURCES
  Test_Hypercube.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/Structure"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainHelpers
  DomainStructure
  Utilities
  )
