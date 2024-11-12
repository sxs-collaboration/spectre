# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY StdHelpers)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Bit.hpp
  RetrieveUniquePtr.hpp
  )
