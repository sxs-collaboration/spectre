# Distributed under the MIT License.
# See LICENSE.txt for details.

# FindCharm.cmake
#
# Finds a suitable Charm++ installation and exposes it via imported targets.
#
# This module requires you have set the `CHARM_ROOT` variable to a Charm++
# installation directory, e.g.:
#
#   cmake -D CHARM_ROOT=/path/to/charm++/build-dir
#
# This is to ensure that SpECTRE builds with the Charm++ installation that you
# intended to use. The `CHARM_ROOT/bin` directory should contain the Charm++
# compiler `charmc`. All information on include directories, library paths,
# compiler flags etc. will be retrieved by invoking `charmc` with the
# `-print-building-blocks` flag. Note that we do _not_ use `charmc` to wrap the
# CMake compiler configuration. Instead, the information retrieved from `charmc`
# is made available to CMake via imported targets.
#
# This module supports CMake "components" to load additional Charm++ modules.
# You can load additional Charm++ modules like this:
#
#   find_package(Charm 6.10.2 EXACT REQUIRED COMPONENTS EveryLB)
#
# This module exposes the following targets:
#
# - `Charmxx::charmxx`: All Charm++ libraries.
# - `Charmxx::pup`: Only the Charm++ PUP serialization library.
# - `Charmxx::main`: Defines the Charm++ main function. Link to compile
#   executables.
# - `CharmModuleInit`: Provides the generated definitions of Charm++'s
#   module-init functions.

option(CHARM_SHARED_LIBS "Link Charm++ shared libraries. Defaults to the \
value of 'BUILD_SHARED_LIBS' but can be overriden to enforce shared or static \
libraries for Charm++." ${BUILD_SHARED_LIBS})

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

# Find the Charm compiler `charmc` first so we can retrieve build info from it
find_program(CHARM_COMPILER
  NAMES charmc
  PATH_SUFFIXES bin
  HINTS ${CHARM_ROOT} ENV CHARM_ROOT
  NO_DEFAULT_PATH
  DOC "The full path to the charm++ compiler 'charmc'"
  )
if(CHARM_COMPILER STREQUAL "CHARM_COMPILER-NOTFOUND")
  message(FATAL_ERROR "The Charm++ compiler 'charmc' could not be found. Make "
    "sure you have set 'CHARM_ROOT' to the path that contains 'bin/charmc'.")
endif()

# Assemble options for invoking `charmc`:
# - Use static or shared libs
if(CHARM_SHARED_LIBS)
  list(APPEND CHARMC_OPTIONS "-charm-shared")
endif()
# - Not linking the main function by default. We expose an extra target that
#   links it because we sometimes want to link Charm++ without a main function.
list(APPEND CHARMC_OPTIONS "-nomain")
list(APPEND CHARMC_OPTIONS "-nomain-module")
# - Request optional Charm++ modules. They can be specified as `COMPONENTS` when
#   calling CMake's `find_package`.
if(Charm_FIND_COMPONENTS)
  list(JOIN Charm_FIND_COMPONENTS "," CHARM_COMPONENTS_JOINED)
  list(APPEND CHARMC_OPTIONS "-modules ${CHARM_COMPONENTS_JOINED}")
endif()

# Invoke 'charmc' so that it outputs build info
list(JOIN CHARMC_OPTIONS " " CHARMC_OPTIONS_JOINED)
set(CHARM_BUILDING_BLOCKS_CMD
  "${CHARM_COMPILER} -print-building-blocks ${CHARMC_OPTIONS_JOINED}")
message(STATUS "Charm++ 'charmc' is invoked as: ${CHARM_BUILDING_BLOCKS_CMD}")
execute_process(
  COMMAND bash -c "${CHARM_BUILDING_BLOCKS_CMD}"
  OUTPUT_VARIABLE CHARM_BUILDING_BLOCKS
  ERROR_VARIABLE CHARM_BUILDING_BLOCKS_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
if(NOT CHARM_BUILDING_BLOCKS)
  if(CHARM_BUILDING_BLOCKS_ERROR MATCHES "lib_so directory not found"
      AND CHARM_SHARED_LIBS)
    message(STATUS "Charm++ is not built with shared-library support so we "
      "fall back to static libraries. Build Charm++ with the '--build-shared' "
      "option to enable shared libraries.")
    set(CHARM_SHARED_LIBS OFF)
    list(REMOVE_ITEM CHARMC_OPTIONS "-charm-shared")
    list(JOIN CHARMC_OPTIONS " " CHARMC_OPTIONS_JOINED)
    set(CHARM_BUILDING_BLOCKS_CMD
      "${CHARM_COMPILER} -print-building-blocks ${CHARMC_OPTIONS_JOINED}")
    execute_process(
      COMMAND bash -c "${CHARM_BUILDING_BLOCKS_CMD}"
      OUTPUT_VARIABLE CHARM_BUILDING_BLOCKS
      ERROR_VARIABLE CHARM_BUILDING_BLOCKS_ERROR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
  else()
    message(FATAL_ERROR "Failed retrieving Charm++ building blocks with "
      "command:\n  ${CHARM_BUILDING_BLOCKS_CMD}\nPlease make sure the Charm++ "
      "version is at least 6.9.0. Error message:\n"
      "${CHARM_BUILDING_BLOCKS_ERROR}")
  endif()
endif()
string(REGEX REPLACE ";" "\\\\;"
  CHARM_BUILDING_BLOCKS_LIST "${CHARM_BUILDING_BLOCKS}")
string(REGEX REPLACE "\n" ";" CHARM_BUILDING_BLOCKS_LIST
  "${CHARM_BUILDING_BLOCKS_LIST}")

# Load the charm building blocks into CMake variables
foreach(CHARM_BUILDING_BLOCK IN LISTS CHARM_BUILDING_BLOCKS_LIST)
  if(CHARM_BUILDING_BLOCK MATCHES "^([^=]+)='(.*)'$")
    set(${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
    string(STRIP "${${CMAKE_MATCH_1}}" ${CMAKE_MATCH_1})
  elseif()
    message(SEND_ERROR "Unexpected output from charmc: ${CHARM_BUILDING_BLOCK}")
  endif()
endforeach()
# Validate the charm building blocks
list(APPEND CHARM_REQUIRED_BUILDING_BLOCKS
  CHARM_CXX_FLAGS
  CHARM_LDXX_FLAGS
  CHARMINC
  # Not needed but available, could be added if needed:
  # CHARM_CC
  # CHARM_CXX
  # CHARM_LD
  # CHARM_LDXX
  # CHARM_CC_FLAGS
  # CHARM_LD_FLAGS
  # CHARMBIN
  )
if(CHARM_SHARED_LIBS)
  list(APPEND CHARM_REQUIRED_BUILDING_BLOCKS CHARMLIBSO)
else()
  list(APPEND CHARM_REQUIRED_BUILDING_BLOCKS CHARMLIB)
endif()
foreach(CHARM_REQUIRED_BUILDING_BLOCK IN LISTS CHARM_REQUIRED_BUILDING_BLOCKS)
  if(NOT ${CHARM_REQUIRED_BUILDING_BLOCK})
    message(FATAL_ERROR "Could not find ${CHARM_REQUIRED_BUILDING_BLOCK} "
      "variable in output of command: ${CHARM_BUILDING_BLOCKS_CMD}")
  endif()
endforeach()

# Split flags into lists
separate_arguments(CHARM_CXX_FLAGS)
separate_arguments(CHARM_LDXX_FLAGS)
# Define variables with standard names for compatibility, though these should
# not be used outside this script.
set(CHARM_INCLUDE_DIR ${CHARMINC})
set(CHARM_INCLUDE_DIRS ${CHARMINC})
if(CHARM_SHARED_LIBS)
  set(CHARM_LIBRARIES ${CHARMLIBSO})
else()
  set(CHARM_LIBRARIES ${CHARMLIB})
endif()

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

# Filter the compiler and linker flags:
# - Remove the macOS deployment target so the CMake setting is used.
list(FILTER CHARM_CXX_FLAGS EXCLUDE REGEX "^-mmacosx-version-min=")
list(FILTER CHARM_LDXX_FLAGS EXCLUDE REGEX "^-mmacosx-version-min=")
# - Remove the C++ standard flag so the CMake setting is used.
list(FILTER CHARM_CXX_FLAGS EXCLUDE REGEX "^-std=")
# - Remove the standard library flag so the CMake setting is used.
list(FILTER CHARM_CXX_FLAGS EXCLUDE REGEX "^-stdlib=")
list(FILTER CHARM_LDXX_FLAGS EXCLUDE REGEX "^-stdlib=")
# - Extract the include directories so we can set them properly with CMake.
#   That's better because they are declared as "SYSTEM", which silences warnings
#   from those headers.
list(FILTER CHARM_CXX_FLAGS EXCLUDE REGEX "^-I${CHARM_INCLUDE_DIR}$")
set(CHARM_INCLUDE_EXTRA ${CHARM_CXX_FLAGS})
list(FILTER CHARM_INCLUDE_EXTRA INCLUDE REGEX "^-I.+")
list(FILTER CHARM_CXX_FLAGS EXCLUDE REGEX "^-I.+")
list(TRANSFORM CHARM_INCLUDE_EXTRA REPLACE "^-I" "")
list(APPEND CHARM_INCLUDE_DIRS ${CHARM_INCLUDE_EXTRA})
# - Remove the rpath linker argument, since CMake adds it automatically.
list(FILTER CHARM_LDXX_FLAGS EXCLUDE REGEX "^-Wl,-rpath,${CHARM_LIBRARIES}/?$")
# - Extract lib directories
list(FILTER CHARM_LDXX_FLAGS EXCLUDE REGEX "^-L${CHARM_LIBRARIES}$")
set(CHARM_LIBRARIES_EXTRA ${CHARM_LDXX_FLAGS})
list(FILTER CHARM_LIBRARIES_EXTRA INCLUDE REGEX "^-L.+")
list(FILTER CHARM_LDXX_FLAGS EXCLUDE REGEX "^-L.+")
list(TRANSFORM CHARM_LIBRARIES_EXTRA REPLACE "^-L" "")
list(APPEND CHARM_LIBRARIES ${CHARM_LIBRARIES_EXTRA})
# - Remove the link directory and find the libraries on the system so we can
#   configure them properly with CMake. That's more robust when switching
#   between static and shared libs builds. Instead of parsing the list of libs
#   below we could just hard-code lib names, but that may break with future
#   Charm versions and could be annoying to debug. However, it would give us
#   more control over which libs to link. Ideally, Charm++ would provide
#   exported targets with these libs.
set(CHARM_LIB_NAMES ${CHARM_LDXX_FLAGS})
list(FILTER CHARM_LIB_NAMES INCLUDE REGEX "^-l.+")
list(FILTER CHARM_LDXX_FLAGS EXCLUDE REGEX "^-l.+")
list(TRANSFORM CHARM_LIB_NAMES REPLACE "^-l" "")
list(JOIN CHARM_LIB_NAMES ", " CHARM_LIB_NAMES_OUTPUT)
message(STATUS "Charm++ lib names: ${CHARM_LIB_NAMES_OUTPUT}")
set(CHARM_LIBS_TO_FIND ${CHARM_LIB_NAMES})
# Also find the 'ckmain' library, which isn't included in the list of lib names
# because we invoked charmc with '-nomain'
list(APPEND CHARM_LIBS_TO_FIND ckmain)
foreach(CHARM_LIB_NAME IN LISTS CHARM_LIBS_TO_FIND)
  find_library(CHARM_LIB_${CHARM_LIB_NAME}
    NAMES ${CHARM_LIB_NAME}
    HINTS ${CHARM_LIBRARIES}
    )
  if ("${CHARM_LIB_${CHARM_LIB_NAME}}"
        STREQUAL "CHARM_LIB_${CHARM_LIB_NAME}-NOTFOUND")
    message(SEND_ERROR "Could not find Charm++ library '${CHARM_LIB_NAME}'. "
      "Make sure you have built Charm++ with support for this library.")
  endif()
  list(APPEND CHARM_LIBS ${CHARM_LIB_${CHARM_LIB_NAME}})
  # Collect libs for the PUP serialization library so we can link it separately
  # if we want. The PUP functionality is in `conv-util` in Charm++ 6.10.2 and
  # in the combined `converse` lib in later versions.
  if(CHARM_LIB_NAME STREQUAL conv-util OR CHARM_LIB_NAME STREQUAL converse)
    list(APPEND PUP_LIBS ${CHARM_LIB_${CHARM_LIB_NAME}})
  endif()
endforeach()
# - Extract libraries that are linked as object files (`conv-static` in Charm++
#   6.10.2).
set(CHARM_LIB_conv-static ${CHARM_LDXX_FLAGS})
list(FILTER CHARM_LIB_conv-static INCLUDE REGEX "conv-static.o$")
list(FILTER CHARM_LDXX_FLAGS EXCLUDE REGEX "conv-static.o$")

# Report remaining flags that will be ignored. They have been used to compile
# Charm++ but may not be compatible with the compiler that the SpECTRE build is
# configured with. They may also include flags that have been chosen differently
# for the SpECTRE build, e.g. optimization settings. The ignored flags are
# printed out to catch cases where adding one of the flags could be useful.
list(JOIN CHARM_CXX_FLAGS " " CHARM_CXX_FLAGS_OUTPUT)
message(STATUS "These Charm++ compiler flags will be ignored: "
  "${CHARM_CXX_FLAGS_OUTPUT}")
list(JOIN CHARM_LDXX_FLAGS " " CHARM_LDXX_FLAGS_OUTPUT)
message(STATUS "These Charm++ linker flags will be ignored: "
  "${CHARM_LDXX_FLAGS_OUTPUT}")

# Invoke `charmc` to generate definitions for the module-init functions.
#
# `charmc` writes a tiny temporary `moduleinit$$.C` file with a few functions
# that often do nothing, compiles it and links it into every Charm module. We
# have `charmc` generate this file and compile it into an object library that
# can be linked into executables. The interface for this could be improved on
# the Charm++ side. See this upstream issue:
# https://github.com/UIUC-PPL/charm/issues/3210
set(CHARM_MODULEINIT_CMD
  "${CHARM_COMPILER} ${CHARMC_OPTIONS_JOINED} -o unused -save")
execute_process(
  COMMAND
  bash -c "(${CHARM_MODULEINIT_CMD} || mv moduleinit*.C CharmModuleInit.C) \
    && rm moduleinit*"
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tmp/
  ERROR_VARIABLE CHARM_MODULEINIT_ERROR
  )
configure_file(
  ${CMAKE_BINARY_DIR}/tmp/CharmModuleInit.C
  ${CMAKE_BINARY_DIR}
  )
add_library(CharmModuleInit OBJECT ${CMAKE_BINARY_DIR}/CharmModuleInit.C)

# Use the charm building blocks to construct imported targets
# - Imported target with all Charm libs
add_library(Charmxx::charmxx INTERFACE IMPORTED)
target_include_directories(Charmxx::charmxx INTERFACE ${CHARM_INCLUDE_DIRS})
target_link_libraries(Charmxx::charmxx INTERFACE ${CHARM_LIBS})
# - Target just for the PUP serialization library.
add_library(Charmxx::pup INTERFACE IMPORTED)
target_include_directories(Charmxx::pup INTERFACE ${CHARM_INCLUDE_DIRS})
target_link_libraries(Charmxx::pup INTERFACE ${PUP_LIBS})
# - Target that defines the Charm++ main function.
add_library(Charmxx::main INTERFACE IMPORTED)
target_link_libraries(
  Charmxx::main
  INTERFACE
  Charmxx::charmxx
  ${CHARM_LIB_ckmain}
  ${CHARM_LIB_conv-static}
  )

# Check if Charm++ was built with SMP support
set(CHARM_SMP NO)
if (EXISTS "${CHARM_INCLUDE_DIR}/conv-mach.h")
  file(READ "${CHARM_INCLUDE_DIR}/conv-mach.h" CONV_MACH_HEADER)
  string(REPLACE "\n" ";" CONV_MACH_HEADER ${CONV_MACH_HEADER})
  foreach(LINE ${CONV_MACH_HEADER})
    if (LINE MATCHES "#define[ \t]+CMK_SMP[ \t]+1")
      set(CHARM_SMP YES)
      break()
    endif()
  endforeach()
else()
  message(SEND_ERROR
    "Unable to detect whether or not Charm++ was built with shared "
    "memory parallelism enabled because the file "
    "'${CHARM_INCLUDE_DIR}/conv-mach.h' was not found. Please file "
    "an issue for support with this error since that file should be "
    "generated as part of the Charm++ build process.")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Charm
  # Only these paths are input variables, i.e. they are searched on the system
  # and can be specified on the command line to select the Charm installation.
  REQUIRED_VARS CHARM_COMPILER
  VERSION_VAR CHARM_VERSION
  )

mark_as_advanced(
  CHARM_COMPILER
  CHARM_SMP
  CHARM_VERSION_MAJOR
  CHARM_VERSION_MINOR
  CHARM_VERSION_PATCH
  CHARM_VERSION
  )
