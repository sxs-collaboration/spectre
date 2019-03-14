# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Convergence)

set(LIBRARY_SOURCES
  Criteria.cpp
  HasConverged.cpp
  Reason.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
