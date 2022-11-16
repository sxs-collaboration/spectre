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
  TensorData.cpp
  VolumeData.cpp
  PYTHON_EXECUTABLES
  ExtractDatFromH5.py
  MODULE_PATH "IO"
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  DataStructures
  H5
  pybind11::module
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  )
