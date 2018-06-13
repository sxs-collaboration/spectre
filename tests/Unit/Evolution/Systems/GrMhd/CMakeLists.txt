# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(ValenciaDivClean)

set(LIBRARY "Test_GrMhd")

set(LIBRARY_SOURCES
  Test_GrMhd.cpp
  )

 add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GrMhd"
  "${LIBRARY_SOURCES}"
  ""
  )
