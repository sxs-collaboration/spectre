# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativitySolutions)

set(LIBRARY_SOURCES
    GaugeWave.cpp
    KerrHorizon.cpp
    KerrSchild.cpp
    Minkowski.cpp
    Tov.cpp
    WrappedGr.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  GeneralRelativity
  Interpolation
  )

add_subdirectory(Python)
