# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EventsAndDenseTriggers)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Filter.hpp
  Or.hpp
  Times.hpp
  )
