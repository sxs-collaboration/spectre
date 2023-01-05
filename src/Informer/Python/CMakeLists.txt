# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyInformer")

spectre_python_add_module(
  Informer
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  InfoAtCompile.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InfoAtCompile.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  Informer
  pybind11::module
  )
