# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GhGrMhdSolutions")

set(LIBRARY_SOURCES
  Test_InstantiateWrappedGr.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  GeneralRelativitySolutions
  GhGrMhdSolutions
  GrMhdSolutions
  Options
  Utilities
  )
