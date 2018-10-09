# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralizedHarmonic)

set(LIBRARY_SOURCES
    Constraints.cpp
    Equations.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
