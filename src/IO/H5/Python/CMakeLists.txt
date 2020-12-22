# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyH5")

spectre_python_add_module(
  H5
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  Dat.cpp
  File.cpp
  VolumeData.cpp
  MODULE_PATH "IO/"
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  IO
  pybind11::module
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  )
