# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_M1GreySolutions")

set(LIBRARY_SOURCES
  Test_ConstantM1.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  M1Grey
  M1GreySolutions
  Utilities
  )
