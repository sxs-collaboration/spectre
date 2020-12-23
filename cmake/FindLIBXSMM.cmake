# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find LIBXSMM: https://github.com/hfp/libxsmm
# If not in one of the default paths specify -D LIBXSMM_ROOT=/path/to/LIBXSMM
# to search there as well.

if(NOT LIBXSMM_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(LIBXSMM_ROOT "")
  set(LIBXSMM_ROOT $ENV{LIBXSMM_ROOT})
endif()

# find the LIBXSMM include directory
find_path(LIBXSMM_INCLUDE_DIRS libxsmm.h
  PATH_SUFFIXES include
  HINTS ${LIBXSMM_ROOT})

find_library(LIBXSMM_LIBRARIES
  NAMES xsmm
  PATH_SUFFIXES lib64 lib
  HINTS ${LIBXSMM_ROOT})

set(LIBXSMM_VERSION "")

function(get_libxsmm_version FILE MAJOR_PREFIX MINOR_PREFIX PATCH_PREFIX)
  if(NOT EXISTS ${FILE})
    return()
  endif()
  # Extract version info from header
  file(READ ${FILE} LIBXSMM_FIND_HEADER_CONTENTS)

  string(REGEX MATCH "${MAJOR_PREFIX}[0-9]+"
    LIBXSMM_MAJOR_VERSION "${LIBXSMM_FIND_HEADER_CONTENTS}")
  if("${LIBXSMM_MAJOR_VERSION}" STREQUAL "")
    return()
  endif()

  string(REPLACE "${MAJOR_PREFIX}" "" LIBXSMM_MAJOR_VERSION
    "${LIBXSMM_MAJOR_VERSION}")

  string(REGEX MATCH "${MINOR_PREFIX}[0-9]+"
    LIBXSMM_MINOR_VERSION "${LIBXSMM_FIND_HEADER_CONTENTS}")
  string(REPLACE "${MINOR_PREFIX}" "" LIBXSMM_MINOR_VERSION
    "${LIBXSMM_MINOR_VERSION}")

  string(REGEX MATCH "${PATCH_PREFIX}[0-9]+"
    LIBXSMM_SUBMINOR_VERSION "${LIBXSMM_FIND_HEADER_CONTENTS}")
  string(REPLACE "${PATCH_PREFIX}" "" LIBXSMM_SUBMINOR_VERSION
    "${LIBXSMM_SUBMINOR_VERSION}")

  set(LIBXSMM_VERSION
    "${LIBXSMM_MAJOR_VERSION}.${LIBXSMM_MINOR_VERSION}.${LIBXSMM_SUBMINOR_VERSION}"
    )
  set(LIBXSMM_VERSION ${LIBXSMM_VERSION} PARENT_SCOPE)
endfunction(get_libxsmm_version FILE MAJOR_REGEX MINOR_REGEX PATCH_REGEX)

get_libxsmm_version(
  ${LIBXSMM_INCLUDE_DIRS}/libxsmm.h
  "#define LIBXSMM_VERSION_MAJOR "
  "#define LIBXSMM_VERSION_MINOR "
  "#define LIBXSMM_VERSION_UPDATE "
  )

get_libxsmm_version(
  ${LIBXSMM_INCLUDE_DIRS}/libxsmm_config.h
  "#define LIBXSMM_CONFIG_VERSION_MAJOR "
  "#define LIBXSMM_CONFIG_VERSION_MINOR "
  "#define LIBXSMM_CONFIG_VERSION_UPDATE "
  )

get_libxsmm_version(
  ${LIBXSMM_INCLUDE_DIRS}/libxsmm_version.h
  "#define LIBXSMM_CONFIG_VERSION_MAJOR "
  "#define LIBXSMM_CONFIG_VERSION_MINOR "
  "#define LIBXSMM_CONFIG_VERSION_UPDATE "
  )

if("${LIBXSMM_VERSION}" STREQUAL "")
  message(WARNING "Failed to detect LIBXSMM version.")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  LIBXSMM
  FOUND_VAR LIBXSMM_FOUND
  REQUIRED_VARS LIBXSMM_INCLUDE_DIRS LIBXSMM_LIBRARIES
  VERSION_VAR LIBXSMM_VERSION)
mark_as_advanced(LIBXSMM_INCLUDE_DIRS LIBXSMM_LIBRARIES
  LIBXSMM_VERSION)
