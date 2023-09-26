# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ExternalIO)

if (NOT TARGET SpEC::Exporter AND NOT TARGET FUKA::Exporter)
  return()
endif()

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

if (TARGET FUKA::Exporter)
  spectre_target_sources(
    ${LIBRARY}
    PRIVATE
    InterpolateFromFuka.cpp
    )
  spectre_target_headers(
    ${LIBRARY}
    INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
    HEADERS
    InterpolateFromFuka.hpp
    )
  target_link_libraries(
    ${LIBRARY}
    PUBLIC
    FUKA::Exporter
    GeneralRelativity
    Hydro
    )
  target_compile_definitions(
    ExternalIO INTERFACE HAS_FUKA_EXPORTER)
endif()

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Utilities
  )
