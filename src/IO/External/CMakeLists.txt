# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ExternalIO)

add_spectre_library(${LIBRARY})

if (TARGET SpEC::Exporter)
  spectre_target_headers(
    ${LIBRARY}
    INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
    HEADERS
    InterpolateFromSpec.hpp
    )
  target_link_libraries(
    ${LIBRARY}
    PUBLIC
    SpEC::Exporter
    )
  target_compile_definitions(
    ExternalIO INTERFACE HAS_SPEC_EXPORTER)
endif()

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Utilities
  )
