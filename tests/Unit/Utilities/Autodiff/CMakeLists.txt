# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Autodiff")

set(LIBRARY_SOURCES
  Test_Autodiff.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Autodiff
  )
