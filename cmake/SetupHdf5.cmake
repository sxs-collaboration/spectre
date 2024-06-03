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

if(NOT TARGET HDF5::HDF5)
  add_library(HDF5::HDF5 INTERFACE IMPORTED)
  set_target_properties(
    HDF5::HDF5
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${HDF5_C_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES}"
    )
  if(DEFINED HDF5_C_DEFINITIONS)
    set_target_properties(
      HDF5::HDF5
      PROPERTIES
      INTERFACE_COMPILE_FLAGS "${HDF5_C_DEFINITIONS}"
      )
  endif()
endif()

if(HDF5_IS_PARALLEL)
  find_package(MPI COMPONENTS C)
  if(MPI_FOUND)
    target_link_libraries(hdf5::hdf5 INTERFACE MPI::MPI_C)
    target_link_libraries(HDF5::HDF5 INTERFACE MPI::MPI_C)
  else()
    message(WARNING "HDF5 is built with MPI support, but MPI was not found. "
      "You may encounter build issues with HDF5, such as missing headers.")
  endif()
endif()

if(HDF5_USE_STATIC_LIBRARIES)
  # If we are using static libraries for HDF5, try to use static libraries
  # for the libsz and transitive libaec dependency. These aren't very
  # common, while other dependencies like libz are, so we leave those
  # as shared library dependencies.
  get_target_property(
    _HDF5_INTERFACE_LINK_LIBS HDF5::HDF5 INTERFACE_LINK_LIBRARIES)
  string(FIND
    "${_HDF5_INTERFACE_LINK_LIBS}" "libsz.so" _LOCATION_OF_LIBSZ)
  if (NOT ${_LOCATION_OF_LIBSZ} EQUAL -1)
    find_library(_libsz NAMES libsz.a)
    if(_libsz)
      string(REPLACE
        "libsz.so" "libsz.a" _HDF5_INTERFACE_LINK_LIBS
        "${_HDF5_INTERFACE_LINK_LIBS}")
      find_library(_libaec NAMES libaec.a)
      if(_libaec)
        list(APPEND _HDF5_INTERFACE_LINK_LIBS ${_libaec})
      endif()
    endif()
  endif()
  string(FIND
    "${_HDF5_INTERFACE_LINK_LIBS}" "libcrypto.so" _LOCATION_OF_LIBCRYPTO)
  if (NOT ${_LOCATION_OF_LIBCRYPTO} EQUAL -1)
    find_library(_libcrypto NAMES libcrypto.a)
    if(_libcrypto)
      string(REPLACE
        "libcrypto.so" "libcrypto.a" _HDF5_INTERFACE_LINK_LIBS
        "${_HDF5_INTERFACE_LINK_LIBS}")
    endif()
  endif()
  string(FIND
    "${_HDF5_INTERFACE_LINK_LIBS}" "libz.so" _LOCATION_OF_LIBZ)
  if (NOT ${_LOCATION_OF_LIBZ} EQUAL -1)
    find_library(_libz NAMES libz.a)
    if(_libz)
      string(REPLACE
        "libz.so" "libz.a" _HDF5_INTERFACE_LINK_LIBS
        "${_HDF5_INTERFACE_LINK_LIBS}")
    endif()
  endif()
  set_target_properties(
    HDF5::HDF5
    PROPERTIES
    INTERFACE_LINK_LIBRARIES "${_HDF5_INTERFACE_LINK_LIBS}"
  )
endif()

# Check if file locking API is available. The versions supporting this feature
# are listed here:
# https://github.com/HDFGroup/hdf5/blob/develop/doc/file-locking.md
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES HDF5::HDF5)
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

include(CheckCXXSourceRuns)
set(CMAKE_REQUIRED_LIBRARIES HDF5::HDF5)
# Logic from src/Utilities/ErrorHandling/FloatingPointExceptions.cpp
check_cxx_source_runs(
  [=[
#include <hdf5.h>

#ifdef __APPLE__
#ifndef __arm64__
#define SPECTRE_FPE_CSR 1
#include <xmmintrin.h>
#endif
#else
#define SPECTRE_FPE_FENV 1
#include <cfenv>
#endif

int main(int /*argc*/, char** /*argv*/) {
#if SPECTRE_FPE_CSR
  _mm_setcsr(_MM_MASK_MASK &
             ~(_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO));
#elif SPECTRE_FPE_FENV
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif
  H5Tcopy(H5T_NATIVE_DOUBLE);
  return 0;
}
]=]
  HDF5_INIT_WITHOUT_FPES
  )
if(NOT HDF5_INIT_WITHOUT_FPES)
  message(FATAL_ERROR
    "The HDF5 library triggers FPEs during initialization.  See upstream bug "
    "https://github.com/HDFGroup/hdf5/issues/3831.  Either switch to an "
    "unaffected version or apply the patch referenced from that issue.")
endif()

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  HDF5::HDF5
  )

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "HDF5 version: ${HDF5_VERSION}\n"
  )

# Find HDF5 tools
get_filename_component(HDF5_TOOLS_DIR HDF5_DIFF_EXECUTABLE DIRECTORY)
find_program(HDF5_REPACK_EXECUTABLE h5repack HINTS ${HDF5_TOOLS_DIR})
