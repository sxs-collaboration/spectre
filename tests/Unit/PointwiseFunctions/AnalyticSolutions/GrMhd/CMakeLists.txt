# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GrMhdSolutions")

set(LIBRARY_SOURCES
  Test_AlfvenWave.cpp
  Test_BondiMichel.cpp
  Test_KomissarovShock.cpp
  Test_SmoothFlow.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  GrMhdSolutions
  Options
  Utilities
  ValenciaDivClean
  )
