# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SystemUtilities)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Abort.cpp
  AttachDebugger.cpp
  Exit.cpp
  ParallelInfo.cpp
  Prefetch.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Abort.hpp
  AttachDebugger.hpp
  Exit.hpp
  ParallelInfo.hpp
  Prefetch.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Charmxx::charmxx
  )
