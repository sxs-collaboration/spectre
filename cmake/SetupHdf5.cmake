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

# Check if file locking API is available. The versions supporting this feature
# are listed here:
# https://github.com/HDFGroup/hdf5/blob/develop/doc/file-locking.md
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES hdf5::hdf5)
check_cxx_source_compiles(
  "#include <hdf5.h>\n\
int main() {\n\
  const hid_t fapl_id = H5Pcopy(H5P_DEFAULT);\n\
  H5Pset_file_locking(fapl_id, false, true);\n\
}"
  HDF5_SUPPORTS_SET_FILE_LOCKING)
if(${HDF5_SUPPORTS_SET_FILE_LOCKING})
  set_property(
    TARGET hdf5::hdf5
    APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
    HDF5_SUPPORTS_SET_FILE_LOCKING)
else()
  message(WARNING "The HDF5 library does not support 'H5Pset_file_locking'. "
    "This means that simulations may crash when you read H5 files while "
    "the simulation is trying to access them. To avoid this, set the "
    "environment variable\n"
    "  HDF5_USE_FILE_LOCKING=FALSE\n"
    "when running simulations, or load an HDF5 module that supports "
    "'H5Pset_file_locking'. Supporting versions are listed here:\n"
    "https://github.com/HDFGroup/hdf5/blob/develop/doc/file-locking.md")
endif()

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  hdf5::hdf5
  )

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "HDF5 version: ${HDF5_VERSION}\n"
  )

# Find HDF5 tools
get_filename_component(HDF5_TOOLS_DIR HDF5_DIFF_EXECUTABLE DIRECTORY)
find_program(HDF5_REPACK_EXECUTABLE h5repack HINTS ${HDF5_TOOLS_DIR})
