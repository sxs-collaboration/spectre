# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(HDF5 REQUIRED C)

message(STATUS "HDF5 libs: " ${HDF5_LIBRARIES})
message(STATUS "HDF5 incl: " ${HDF5_INCLUDE_DIRS})
message(STATUS "HDF5 vers: " ${HDF5_VERSION})

add_library(Hdf5 INTERFACE IMPORTED)
set_property(TARGET Hdf5
  APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS})
set_property(TARGET Hdf5
  APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${HDF5_LIBRARIES})

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Hdf5
  )

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "HDF5 Version:  ${HDF5_VERSION}\n"
  )
