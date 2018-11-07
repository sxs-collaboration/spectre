# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativitySolutions)

set(LIBRARY_SOURCES
    KerrHorizon.cpp
    KerrSchild.cpp
    Minkowski.cpp
    Tov.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE GeneralRelativity
  INTERFACE Interpolation
  )
