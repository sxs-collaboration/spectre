# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyInformer")

spectre_python_add_module(
  Informer
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  InfoAtCompile.cpp
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  Informer
  pybind11::module
  )
