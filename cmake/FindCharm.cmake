# Distributed under the MIT License.
# See LICENSE.txt for details.

if (DEFINED ENV{CHARM_ROOT} AND "${CHARM_ROOT}" STREQUAL "")
  set(CHARM_ROOT "$ENV{CHARM_ROOT}")
endif()

if (NOT EXISTS "${CHARM_ROOT}")
  if ("${CHARM_ROOT}" STREQUAL "")
    message(
        FATAL_ERROR "CHARM_ROOT was not set. Pass it as a command-line arg: "
        "cmake -D CHARM_ROOT=/path/to/charm++/build-dir")
  endif()
  message(
      FATAL_ERROR "CHARM_ROOT=${CHARM_ROOT} does not exist. "
      "Please pass it as a command-line definition to cmake, i.e. "
      "cmake -D CHARM_ROOT=/path/to/charm++/build-dir"
  )
endif ()

find_path(
    CHARM_INCLUDE_DIRS charm.h
    PATH_SUFFIXES include
    HINTS ${CHARM_ROOT}
)

if (EXISTS "${CHARM_INCLUDE_DIRS}/VERSION")
  set(CHARM_VERSION_FILE_LOCATION "${CHARM_INCLUDE_DIRS}/VERSION")
elseif(EXISTS "${CHARM_ROOT}/VERSION")
  set(CHARM_VERSION_FILE_LOCATION "${CHARM_ROOT}/VERSION")
else()
  message(FATAL_ERROR "Failed to find Charm++ version file")
endif()

file(READ "${CHARM_VERSION_FILE_LOCATION}" CHARM_VERSION_FILE)
string(REGEX REPLACE "\n" "" CHARM_VERSION_FILE "${CHARM_VERSION_FILE}")
STRING(
    REGEX REPLACE
    "([0-9])0([0-9])0([0-9])"
    "\\1;\\2;\\3"
    CHARM_VERSIONS_PARSED
    ${CHARM_VERSION_FILE}
)

list(GET CHARM_VERSIONS_PARSED 0 CHARM_MAJOR_VERSION)
list(GET CHARM_VERSIONS_PARSED 1 CHARM_MINOR_VERSION)
list(GET CHARM_VERSIONS_PARSED 2 CHARM_PATCH_VERSION)
set(CHARM_VERSION
  "${CHARM_MAJOR_VERSION}.${CHARM_MINOR_VERSION}.${CHARM_PATCH_VERSION}")

find_library(CHARM_LIBCK
    NAMES ck
    PATH_SUFFIXES lib
    HINTS ${CHARM_ROOT}
)
get_filename_component(CHARM_LIBRARIES ${CHARM_LIBCK} DIRECTORY)

find_program(CHARM_COMPILER
  NAMES charmc
  PATH_SUFFIXES bin
  HINTS ${CHARM_ROOT}
  DOC "The full-path to the charm++ compiler"
  )

# Handle the QUIETLY and REQUIRED arguments and set CHARM_FOUND to TRUE if all
# listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Charm
  FOUND_VAR CHARM_FOUND
  REQUIRED_VARS CHARM_COMPILER CHARM_INCLUDE_DIRS CHARM_LIBRARIES
  VERSION_VAR CHARM_VERSION
  )

mark_as_advanced(
    CHARM_COMPILER
    CHARM_INCLUDE_DIRS
    CHARM_MAJOR_VERSION
    CHARM_MINOR_VERSION
    CHARM_PATCH_VERSION
    CHARM_VERSION
    CHARM_LIBRARIES
)
