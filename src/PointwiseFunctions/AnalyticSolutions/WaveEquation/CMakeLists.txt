# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY WaveEquationSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  PlaneWave.cpp
  RegularSphericalWave.cpp
  SemidiscretizedDg.cpp
  StandingWave.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  PlaneWave.hpp
  RegularSphericalWave.hpp
  SemidiscretizedDg.hpp
  Solutions.hpp
  StandingWave.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  Serialization
  )
