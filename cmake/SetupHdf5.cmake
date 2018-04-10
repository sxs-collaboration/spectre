# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(HDF5 REQUIRED C)

message(STATUS "HDF5 libs: " ${HDF5_LIBRARIES})
message(STATUS "HDF5 incl: " ${HDF5_INCLUDE_DIRS})
message(STATUS "HDF5 vers: " ${HDF5_VERSION})
spectre_include_directories(${HDF5_INCLUDE_DIRS})
list(APPEND SPECTRE_LIBRARIES ${HDF5_LIBRARIES})

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "HDF5 Version:  ${HDF5_VERSION}\n"
  )
