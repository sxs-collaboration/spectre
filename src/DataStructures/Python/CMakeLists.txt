# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyDataStructures")

spectre_python_add_module(
  DataStructures
  SOURCES
  Bindings.cpp
  DataVector.cpp
  Matrix.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  ${SPECTRE_LINK_PYBINDINGS}
  )
