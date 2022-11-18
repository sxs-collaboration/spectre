# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "TimeStepperHelpers")

set(LIBRARY_SOURCES
  RungeKutta.cpp
  TimeStepperTestUtils.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Time
  Utilities
  PRIVATE
  ErrorHandling
  Framework
  )
