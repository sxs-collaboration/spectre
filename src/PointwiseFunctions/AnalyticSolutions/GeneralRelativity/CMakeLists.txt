# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativitySolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  GaugeWave.cpp
  KerrHorizon.cpp
  KerrSchild.cpp
  Minkowski.cpp
  Tov.cpp
  WrappedGr.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  GaugeWave.hpp
  KerrHorizon.hpp
  KerrSchild.hpp
  Minkowski.hpp
  Solutions.hpp
  Tov.hpp
  WrappedGr.hpp
  WrappedGr.tpp
  )

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
