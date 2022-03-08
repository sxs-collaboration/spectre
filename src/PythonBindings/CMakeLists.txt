# Distributed under the MIT License.
# See LICENSE.txt for details.

if (${BUILD_PYTHON_BINDINGS})
  set(LIBRARY PythonBindings)

  add_spectre_library(${LIBRARY})

  spectre_target_sources(
    ${LIBRARY}
    PRIVATE
    CharmCompatibility.cpp
    FormalineCompatibility.cpp
    InfoAtLinkCompatibility.cpp
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
    PUBLIC
    Utilities
    )
endif()

# Generate shell completion scripts. These don't usually change, so they are
# committed to the source code and configured to the build dir. See the
# [click docs](https://click.palletsprojects.com/shell-completion/)
# for instructions how to regenerate the shell completion scripts.
configure_file(
  shell-completion.bash
  ${SPECTRE_PYTHON_PREFIX_PARENT}/shell-completion.bash @ONLY)
configure_file(
  shell-completion.fish
  ${SPECTRE_PYTHON_PREFIX_PARENT}/shell-completion.fish @ONLY)
configure_file(
  shell-completion.zsh
  ${SPECTRE_PYTHON_PREFIX_PARENT}/shell-completion.zsh @ONLY)
