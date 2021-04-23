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
    CHARM_INCLUDE_DIR charm.h
    PATH_SUFFIXES include
    HINTS ${CHARM_ROOT} ENV CHARM_ROOT
)
set(CHARM_INCLUDE_DIRS ${CHARM_INCLUDE_DIR})

# Find version file
if(EXISTS "${CHARM_INCLUDE_DIR}/charm-version.h")
  set(CHARM_VERSION_FILE_VERSION "6_11")
  set(CHARM_VERSION_FILE_LOCATION "${CHARM_INCLUDE_DIR}/charm-version.h")
elseif(EXISTS "${CHARM_INCLUDE_DIR}/VERSION")
  set(CHARM_VERSION_FILE_VERSION "pre_6_11")
  set(CHARM_VERSION_FILE_LOCATION "${CHARM_INCLUDE_DIR}/VERSION")
elseif(EXISTS "${CHARM_ROOT}/VERSION")
  set(CHARM_VERSION_FILE_VERSION "pre_6_11")
  set(CHARM_VERSION_FILE_LOCATION "${CHARM_ROOT}/VERSION")
else()
  message(FATAL_ERROR "Failed to find Charm++ version file")
endif()

# Parse version from file
file(READ "${CHARM_VERSION_FILE_LOCATION}" CHARM_VERSION_FILE)
if(CHARM_VERSION_FILE_VERSION STREQUAL "6_11")
  # Since version 6.11 the file is C++-compatible
  if(CHARM_VERSION_FILE MATCHES "#define CHARM_VERSION_MAJOR ([0-9]+)")
    set(CHARM_VERSION_MAJOR ${CMAKE_MATCH_1})
  else()
    message(FATAL_ERROR "Could not parse CHARM_VERSION_MAJOR from file: "
      "${CHARM_VERSION_FILE_LOCATION}")
  endif()
  if(CHARM_VERSION_FILE MATCHES "#define CHARM_VERSION_MINOR ([0-9]+)")
    set(CHARM_VERSION_MINOR ${CMAKE_MATCH_1})
  else()
    message(FATAL_ERROR "Could not parse CHARM_VERSION_MINOR from file: "
      "${CHARM_VERSION_FILE_LOCATION}")
  endif()
  if(CHARM_VERSION_FILE MATCHES "#define CHARM_VERSION_PATCH ([0-9]+)")
    set(CHARM_VERSION_PATCH ${CMAKE_MATCH_1})
  else()
    message(FATAL_ERROR "Could not parse CHARM_VERSION_PATCH from file: "
      "${CHARM_VERSION_FILE_LOCATION}")
  endif()
elseif(CHARM_VERSION_FILE_VERSION STREQUAL "pre_6_11")
  # Before version 6.11 the file contains only a string
  string(REGEX REPLACE "\n" "" CHARM_VERSION_FILE "${CHARM_VERSION_FILE}")
  string(
    REGEX REPLACE
    "([0-9])1([0-9])0([0-9])"
    "\\1;1\\2;\\3"
    CHARM_VERSIONS_PARSED
    ${CHARM_VERSION_FILE}
    )
  list(GET CHARM_VERSIONS_PARSED 0 CHARM_VERSION_MAJOR)
  list(GET CHARM_VERSIONS_PARSED 1 CHARM_VERSION_MINOR)
  list(GET CHARM_VERSIONS_PARSED 2 CHARM_VERSION_PATCH)
endif()
set(CHARM_VERSION
  "${CHARM_VERSION_MAJOR}.${CHARM_VERSION_MINOR}.${CHARM_VERSION_PATCH}")

find_library(CHARM_LIBCK
    NAMES ck
    PATH_SUFFIXES lib
    HINTS ${CHARM_ROOT} ENV CHARM_ROOT
)

get_filename_component(CHARM_LIBRARIES ${CHARM_LIBCK} DIRECTORY)

find_library(CHARM_EVERYLB
  NAMES moduleEveryLB
  HINTS ${CHARM_LIBRARIES}
  NO_DEFAULT_PATH
  )

if("${CHARM_EVERYLB}" STREQUAL "CHARM_EVERYLB-NOTFOUND")
  message(SEND_ERROR "Could not find charm module EveryLB. "
    "Make sure you have built the LIBS target when building charm++")
endif()

find_program(CHARM_COMPILER
  NAMES charmc
  PATH_SUFFIXES bin
  HINTS ${CHARM_ROOT} ENV CHARM_ROOT
  DOC "The full-path to the charm++ compiler"
  )

# Handle the QUIETLY and REQUIRED arguments and set CHARM_FOUND to TRUE if all
# listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Charm
  REQUIRED_VARS CHARM_COMPILER CHARM_INCLUDE_DIR CHARM_LIBCK
  VERSION_VAR CHARM_VERSION
  )

mark_as_advanced(
  CHARM_COMPILER
  CHARM_INCLUDE_DIR
  CHARM_LIBCK
  CHARM_EVERYLB
  )
