# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SystemUtilities)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ParallelInfo.hpp
  )
