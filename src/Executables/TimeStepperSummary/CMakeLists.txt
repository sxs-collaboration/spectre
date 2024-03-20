# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE TimeStepperSummary)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  TimeStepperSummary.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Boost::boost
  Boost::program_options
  ErrorHandling
  Printf
  Time
  Utilities
  )
