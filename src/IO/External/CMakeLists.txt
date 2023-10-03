# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ExternalIO)

set(_LIB_TYPE INTERFACE)
if (TARGET FUKA::Exporter)
  set(_LIB_TYPE "")
endif()

add_spectre_library(${LIBRARY} ${_LIB_TYPE})

if (TARGET SpEC::Exporter)
  spectre_target_headers(
    ${LIBRARY}
    INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
    HEADERS
    InterpolateFromSpec.hpp
    )
  target_link_libraries(
    ${LIBRARY}
    INTERFACE
    DataStructures
    ErrorHandling
    SpEC::Exporter
    Utilities
    )
  target_compile_definitions(
    ${LIBRARY} INTERFACE HAS_SPEC_EXPORTER)
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
    DataStructures
    FUKA::Exporter
    GeneralRelativity
    Hydro
    Utilities
    PRIVATE
    ErrorHandling
    )
  target_compile_definitions(
    ${LIBRARY} INTERFACE HAS_FUKA_EXPORTER)
endif()
