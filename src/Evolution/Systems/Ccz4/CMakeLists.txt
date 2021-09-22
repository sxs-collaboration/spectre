# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Ccz4)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Christoffel.cpp
  DerivLapse.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Christoffel.hpp
  DerivLapse.hpp
  Tags.hpp
  TagsDeclarations.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  GeneralRelativity
  Options
  Utilities
  INTERFACE
  )
