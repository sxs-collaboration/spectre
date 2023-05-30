# Distributed under the MIT License.
# See LICENSE.txt for details.

# Setup Sphinx to generate documentation for our Python modules. The HTML output
# is placed in docs/html/py/ so we can link to it from Doxygen.

# We have also experimented with generating our C++ documentation with Sphinx.
# See issue: https://github.com/sxs-collaboration/spectre/issues/2138

set(SPHINX_SOURCE ${CMAKE_BINARY_DIR}/docs/sphinx)
set(SPHINX_BUILD ${CMAKE_BINARY_DIR}/docs/html/py)

configure_file(
  ${CMAKE_SOURCE_DIR}/docs/conf.py
  ${SPHINX_SOURCE}/conf.py
  @ONLY
  )

configure_file(
  ${CMAKE_SOURCE_DIR}/docs/index.rst
  ${SPHINX_SOURCE}/index.rst
  )

file(
  COPY ${CMAKE_SOURCE_DIR}/docs/_templates
  DESTINATION ${SPHINX_SOURCE}
)

add_custom_target(py-docs ALL
  COMMAND
  ${CMAKE_COMMAND} -E env
  ${CMAKE_BINARY_DIR}/bin/python-spectre -m sphinx -b html
  ${SPHINX_SOURCE} ${SPHINX_BUILD}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating Python documentation"
  )
add_dependencies(py-docs all-pybindings)
