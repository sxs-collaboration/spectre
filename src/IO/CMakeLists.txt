# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY IO)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Connectivity.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Connectivity.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost # PUBLIC because of the actions
  DataStructures
  ErrorHandling
  Hdf5
  Options
  Utilities
  INTERFACE
  Domain
  DomainStructure
  )

add_subdirectory(H5)
add_subdirectory(Importers)
add_subdirectory(Observer)
