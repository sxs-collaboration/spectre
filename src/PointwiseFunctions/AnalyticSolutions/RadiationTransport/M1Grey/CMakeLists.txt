# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY M1GreySolutions)

set(LIBRARY_SOURCES
    ConstantM1.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  INTERFACE GeneralRelativitySolutions
  INTERFACE Hydro
  INTERFACE M1Grey
  )
