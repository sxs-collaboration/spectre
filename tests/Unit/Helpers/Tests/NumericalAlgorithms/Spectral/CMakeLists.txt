# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_SpectralHelpers")

set(LIBRARY_SOURCES
  Test_BallTestFunctions.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  SpectralHelpers
  )
