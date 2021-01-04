# Distributed under the MIT License.
# See LICENSE.txt for details.

if (${BUILD_PYTHON_BINDINGS})
  set(LIBRARY PyBindings)

  add_spectre_library(${LIBRARY})

  spectre_target_sources(
    ${LIBRARY}
    PRIVATE
    CharmCompatibility.cpp
    InfoAtLink.cpp
    )

  spectre_target_headers(
    ${LIBRARY}
    INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
    HEADERS
    BoundChecks.hpp
    )

  target_link_libraries(
    ${LIBRARY}
    PRIVATE
    Informer
    INTERFACE
    Utilities
    )
endif()
