# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_H5")

set(LIBRARY_SOURCES
  Test_CheckH5PropertiesMatch.cpp
  Test_Dat.cpp
  Test_EosTable.cpp
  Test_H5.cpp
  Test_H5File.cpp
  Test_OpenGroup.cpp
  Test_StellarCollapseEos.cpp
  Test_TensorData.cpp
  Test_Version.cpp
  Test_VolumeData.cpp
  )

add_test_library(
  ${LIBRARY}
  "IO/H5/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  DataStructures
  Domain
  DomainCreators
  H5
  Informer
  IO
  IoTestHelpers
  Parallel
  Spectral
  Utilities
  )

spectre_add_python_bindings_test(
  "Unit.IO.H5.Python"
  "Test_H5.py"
  "unit;IO;H5;python"
  PyH5
  )

spectre_add_python_bindings_test(
  "Unit.IO.H5.VolumeData.Python"
  "Test_VolumeData.py"
  "unit;IO;H5;Python"
  PyH5
  )

spectre_add_python_bindings_test(
  "Unit.IO.H5.Python.TensorData"
  "Test_TensorData.py"
  "unit;IO;H5;Python"
  PyH5
  )

add_subdirectory(Python)
