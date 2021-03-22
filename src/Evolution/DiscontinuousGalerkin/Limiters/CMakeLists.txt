# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Limiters)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  HwenoImpl.cpp
  MinmodHelpers.cpp
  MinmodImpl.cpp
  MinmodTci.cpp
  MinmodType.cpp
  WenoGridHelpers.cpp
  WenoHelpers.cpp
  WenoOscillationIndicator.cpp
  WenoType.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  HwenoImpl.hpp
  Krivodonova.hpp
  LimiterActions.hpp
  Limiters.hpp
  Minmod.hpp
  MinmodHelpers.hpp
  MinmodImpl.hpp
  MinmodTci.hpp
  MinmodType.hpp
  SimpleWenoImpl.hpp
  Tags.hpp
  Tci.hpp
  Weno.hpp
  WenoGridHelpers.hpp
  WenoHelpers.hpp
  WenoOscillationIndicator.hpp
  WenoType.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  Domain
  DomainStructure
  Interpolation
  LinearOperators
  Options
  Spectral
  )
