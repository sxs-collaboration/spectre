# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyInformer")

spectre_python_add_module(
  Informer
  SOURCES
  Bindings.cpp
  InfoAtCompile.cpp
)

target_link_libraries(
  ${LIBRARY}
  PUBLIC Informer
  ${SPECTRE_LINK_PYBINDINGS}
  )
