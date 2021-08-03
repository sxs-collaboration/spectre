# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ValenciaDivCleanLimiters)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  VariablesToLimit.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  VariablesToLimit.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  Domain
  ErrorHandling
  Limiters
  ValenciaDivClean
  Utilities
  )
