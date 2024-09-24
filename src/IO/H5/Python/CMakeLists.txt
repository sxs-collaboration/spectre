# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyH5")

spectre_python_add_module(
  H5
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  Cce.cpp
  CombineH5.cpp
  Dat.cpp
  File.cpp
  TensorData.cpp
  VolumeData.cpp
  PYTHON_FILES
  __init__.py
  CombineH5.py
  CombineH5Dat.py
  DeleteSubfiles.py
  ExtendConnectivityData.py
  ExtractDatFromH5.py
  ExtractInputSourceYamlFromH5.py
  IterElements.py
  MODULE_PATH "IO"
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Cce.hpp
  CombineH5.hpp
  Dat.hpp
  File.hpp
  TensorData.hpp
  VolumeData.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  DomainCreators
  DataStructures
  H5
  pybind11::module
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PySpectral
  )
