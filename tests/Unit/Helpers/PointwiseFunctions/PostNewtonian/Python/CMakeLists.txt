# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyPostNewtonianHelpers")

spectre_python_add_module(
  PostNewtonian
  MODULE_PATH "testing"
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  PYTHON_FILES
  __init__.py
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  PostNewtonianHelpers
  pybind11::module
  )
