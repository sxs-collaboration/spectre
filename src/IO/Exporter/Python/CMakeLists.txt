# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyExporter")

spectre_python_add_module(
  Exporter
  MODULE_PATH "IO"
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  PYTHON_FILES
  __init__.py
  InterpolateToPoints.py
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  ErrorHandling
  Exporter
  pybind11::module
  Utilities
  )
