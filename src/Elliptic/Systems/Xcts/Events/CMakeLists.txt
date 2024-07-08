# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY XctsEvents)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ObserveAdmIntegrals.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ObserveAdmIntegrals.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  Elliptic
  Xcts
  Options
  Serialization
  Utilities
  XctsPointwiseFunctions
  )
