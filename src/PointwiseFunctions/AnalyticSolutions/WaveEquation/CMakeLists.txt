# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY WaveEquation)

set(LIBRARY_SOURCES
  PlaneWave.cpp
  )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE MathFunctions
  )
