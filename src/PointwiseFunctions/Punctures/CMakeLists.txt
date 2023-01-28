# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PuncturesPointwiseFunctions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AdmIntegrals.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AdmIntegrals.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Punctures
  Utilities
  )

add_subdirectory(Python)
