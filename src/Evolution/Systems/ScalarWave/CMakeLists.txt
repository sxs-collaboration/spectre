# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarWave)

set(LIBRARY_SOURCES
  Characteristics.cpp
  Constraints.cpp
  Equations.cpp
  UpwindPenaltyCorrection.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  )
