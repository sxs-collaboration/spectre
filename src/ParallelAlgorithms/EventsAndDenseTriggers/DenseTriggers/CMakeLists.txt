# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EventsAndDenseTriggers)

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Filter.cpp
  Or.cpp
  Times.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  Filter.hpp
  Or.hpp
  Times.hpp
  )
