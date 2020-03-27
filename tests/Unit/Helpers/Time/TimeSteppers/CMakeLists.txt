# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "TimeStepperHelpers")

set(LIBRARY_SOURCES
  TimeStepperTestUtils.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ErrorHandling
  Time
  Utilities
  )
