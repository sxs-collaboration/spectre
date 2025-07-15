# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativitySolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Ccz4WrappedGr.cpp
  GaugeWave.cpp
  GaugePlaneWave.cpp
  HarmonicSchwarzschild.cpp
  KerrSchild.cpp
  Minkowski.cpp
  SphericalKerrSchild.cpp
  TrumpetSchwarzschild.cpp
  WrappedGr.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Ccz4WrappedGr.hpp
  Ccz4WrappedGr.tpp
  Factory.hpp
  GaugeWave.hpp
  GaugePlaneWave.hpp
  HarmonicSchwarzschild.hpp
  KerrSchild.hpp
  Minkowski.hpp
  Solutions.hpp
  SphericalKerrSchild.hpp
  TrumpetSchwarzschild.hpp
  WrappedGr.hpp
  WrappedGr.tpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  Ccz4
  DataStructures
  ErrorHandling
  GeneralRelativity
  Interpolation
  Serialization
  SpecialRelativity
  )

add_subdirectory(Python)
