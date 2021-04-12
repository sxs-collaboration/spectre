# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelAmr")

set(LIBRARY_SOURCES
  Actions/Test_Initialize.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/Amr"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Amr
  DomainStructure
  Utilities
  )
