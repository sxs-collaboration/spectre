# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticUtilities")

set(LIBRARY_SOURCES
  Test_ApplyAt.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainStructure
  Elliptic
  Utilities
  )
