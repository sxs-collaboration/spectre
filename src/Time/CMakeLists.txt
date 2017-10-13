# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Time)

set(LIBRARY_SOURCES
    Slab.cpp
    Time.cpp
    TimeId.cpp
    )

add_subdirectory(TimeSteppers)

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE ErrorHandling
  )
