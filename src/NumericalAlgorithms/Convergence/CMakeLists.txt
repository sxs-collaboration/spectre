# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Convergence)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Criteria.cpp
  HasConverged.cpp
  Reason.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Convergence.hpp
  Criteria.hpp
  HasConverged.hpp
  Reason.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  Options
  )
