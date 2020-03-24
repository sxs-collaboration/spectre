# Distributed under the MIT License.
# See LICENSE.txt for details.


# We cannot fully support Sphinx+Breathe until Sphinx version 3.0.0. We have
# an issue sxs-collaboration/spectre#2138 for tracking Sphinx bugs that
# block us from being able to use Sphinx+Breathe.
#
# Some notes:
# - Currently one must manually run make doc-xml before running Sphinx
#   in order to get the Doxygen XML output. We will want to automate
#   this in the future with true tracking of files needed for building
#   Doxygen and Sphinx. Doing so will reduce the time it takes to rebuild
#   documentation.
# - Breathe's XML parser is horribly slow. See sxs-collaboration/spectre#2138
option(USE_SPHINX
  "When enabled, find and set up Sphinx+Breathe for documentation"
  OFF)

if (DOXYGEN_FOUND AND USE_SPHINX)
  find_package(Sphinx REQUIRED)
  find_package(Breathe REQUIRED)

  configure_file(
    ${CMAKE_SOURCE_DIR}/docs/conf.py
    ${CMAKE_BINARY_DIR}/docs/conf.py
    @ONLY
    )

  configure_file(
    ${CMAKE_SOURCE_DIR}/docs/index.rst
    ${CMAKE_BINARY_DIR}/docs/index.rst
    )

  set(SPHINX_SOURCE ${CMAKE_BINARY_DIR}/docs)
  set(SPHINX_BUILD ${CMAKE_BINARY_DIR}/docs/sphinx)
  set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

  add_custom_target(Sphinx ALL
    COMMAND
    ${SPHINX_EXECUTABLE} -b html
    ${SPHINX_SOURCE} ${SPHINX_BUILD}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating documentation with Sphinx"
    )
endif(DOXYGEN_FOUND AND USE_SPHINX)
