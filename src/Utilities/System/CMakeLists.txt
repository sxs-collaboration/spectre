# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SystemUtilities)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Abort.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Abort.hpp
  Exit.hpp
  ParallelInfo.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Charmxx::charmxx
  )
