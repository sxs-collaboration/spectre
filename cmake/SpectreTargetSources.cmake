# Distributed under the MIT License.
# See LICENSE.txt for details.

# Add sources to a target.
#
# Before CMake v3.13 CMake's target_sources always found files relative to
# the target's directory, which is rather unnatural while in subdirectories.
# This wrapper function provides the new CMake behavior for older versions
# of CMake.
#
# Usage:
#   spectre_target_sources(
#     TARGET
#     PRIVATE
#     A.cpp
#     B.cpp
#     C.cpp
#     )
#
# Note that unless you know for sure a source file should be PUBLIC or INTERFACE
# just mark it PRIVATE. A particular source file should be marked
# PUBLIC/INTERFACE only if it should also be built by consuming targets.
#
# Arguments:
# - TARGET: the name of the target to which to add the sources.
# - EXTRA_ARGS: the sources, PUBLIC/PRIVATE/INTERFACE keywords, and generator
#               expressions to set the sources to.
function(spectre_target_sources TARGET)
  if(POLICY CMP0076)
    # New behavior is available, so just forward to it by ensuring
    # that we have the policy set to request the new behavior, but
    # don't change the policy setting for the calling scope
    cmake_policy(PUSH)
    cmake_policy(SET CMP0076 NEW)
    target_sources(${TARGET} ${ARGN})
    cmake_policy(POP)
    return()
  endif()

  # Must be using CMake 3.12 or earlier, so simulate the new behavior
  unset(_SOURCE_FILES)
  get_target_property(_TARGET_SOURCE_DIR ${TARGET} SOURCE_DIR)

  foreach(SOURCE_FILE ${ARGN})
    string(FIND "${SOURCE_FILE}" "$<" GENERATOR_EXPR_START_INDEX)
    set(_IS_GENERATOR_EXPR OFF)
    if(${GENERATOR_EXPR_START_INDEX} EQUAL 0)
      set(_IS_GENERATOR_EXPR ON)
    endif(${GENERATOR_EXPR_START_INDEX} EQUAL 0)

    if(NOT SOURCE_FILE STREQUAL "PRIVATE" AND
        NOT SOURCE_FILE STREQUAL "PUBLIC" AND
        NOT SOURCE_FILE STREQUAL "INTERFACE" AND
        NOT IS_ABSOLUTE "${SOURCE_FILE}" AND
        NOT _IS_GENERATOR_EXPR)
      # Absolute path to source
      set(SOURCE_FILE "${CMAKE_CURRENT_LIST_DIR}/${SOURCE_FILE}")
    endif()
    list(APPEND _SOURCE_FILES ${SOURCE_FILE})
  endforeach()
  target_sources(${TARGET} ${_SOURCE_FILES})
endfunction()
