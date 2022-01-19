# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(HDF5 REQUIRED COMPONENTS C)

message(STATUS "HDF5 libs: " ${HDF5_C_LIBRARIES})
message(STATUS "HDF5 incl: " ${HDF5_C_INCLUDE_DIRS})
message(STATUS "HDF5 vers: " ${HDF5_VERSION})

if(NOT TARGET hdf5::hdf5)
  add_library(hdf5::hdf5 INTERFACE IMPORTED)
  set_target_properties(
    hdf5::hdf5
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${HDF5_C_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES}"
    )
  if(DEFINED HDF5_C_DEFINITIONS)
    set_target_properties(
      hdf5::hdf5
      PROPERTIES
      INTERFACE_COMPILE_FLAGS "${HDF5_C_DEFINITIONS}"
      )
  endif()
endif()

if(HDF5_IS_PARALLEL)
  find_package(MPI COMPONENTS C)
  if(MPI_FOUND)
    target_link_libraries(hdf5::hdf5 INTERFACE MPI::MPI_C)
  else()
    message(WARNING "HDF5 is built with MPI support, but MPI was not found. "
      "You may encounter build issues with HDF5, such as missing headers.")
  endif()
endif()

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  hdf5::hdf5
  )

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "HDF5 version: ${HDF5_VERSION}\n"
  )
