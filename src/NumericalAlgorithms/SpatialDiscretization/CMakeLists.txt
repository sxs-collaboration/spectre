# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SpatialDiscretization)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Basis.cpp
  Mesh.cpp
  Quadrature.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Basis.hpp
  Mesh.hpp
  OptionTags.hpp
  Quadrature.hpp
  SpatialDiscretization.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  Utilities
  PRIVATE
  Charmxx::pup
 )
