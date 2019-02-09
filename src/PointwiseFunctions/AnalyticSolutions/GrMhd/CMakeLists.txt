# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrMhdSolutions)

set(LIBRARY_SOURCES
    AlfvenWave.cpp
    BondiMichel.cpp
    KomissarovShock.cpp
    SmoothFlow.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE GeneralRelativitySolutions
  INTERFACE Hydro
  )
