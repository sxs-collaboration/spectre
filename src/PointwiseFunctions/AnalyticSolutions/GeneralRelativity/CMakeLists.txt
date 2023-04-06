# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativitySolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  GaugeWave.cpp
  HarmonicSchwarzschild.cpp
  KerrHorizon.cpp
  KerrSchild.cpp
  Minkowski.cpp
  SphericalKerrSchild.cpp
  WrappedGr.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  GaugeWave.hpp
  HarmonicSchwarzschild.hpp
  KerrHorizon.hpp
  KerrSchild.hpp
  Minkowski.hpp
  Solutions.hpp
  SphericalKerrSchild.hpp
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
  Serialization
  )
