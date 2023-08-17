# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Ader")

set(LIBRARY_SOURCES
  Test_Matrices.cpp
  )

add_test_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Ader
  DataStructures
  ErrorHandling
  Spectral
  )
