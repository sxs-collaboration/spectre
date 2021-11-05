# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY FiniteDifference)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AoWeno.cpp
  Minmod.cpp
  MonotisedCentral.cpp
  Unlimited.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AoWeno.hpp
  FiniteDifference.hpp
  Minmod.hpp
  MonotisedCentral.hpp
  Reconstruct.hpp
  Reconstruct.tpp
  Unlimited.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Utilities
  PRIVATE
  DataStructures
  DomainStructure
  ErrorHandling
  )
