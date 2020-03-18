# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Time)

set(LIBRARY_SOURCES
    Slab.cpp
    Time.cpp
    TimeStepId.cpp
    Utilities.cpp
    )

add_subdirectory(StepChoosers)
add_subdirectory(StepControllers)
add_subdirectory(TimeSteppers)

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  ErrorHandling
  Utilities
  )
