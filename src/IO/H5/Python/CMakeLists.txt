# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyH5")

spectre_python_add_module(
  H5
  SOURCES
  Bindings.cpp
  Dat.cpp
  File.cpp
  VolumeData.cpp
  MODULE_PATH "IO/"
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC IO
  PUBLIC DataStructures
  PUBLIC Informer
  ${SPECTRE_LINK_PYBINDINGS}
  )

add_dependencies(
  ${LIBRARY}
  PyDataStructures
  )
