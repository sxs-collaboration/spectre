# Distributed under the MIT License.
# See LICENSE.txt for details.

# Checks if a CXX flag is supported by the compiler and adds it if it is
# - FLAG_TO_CHECK: the CXX flag to add if the compiler supports it
function(check_and_add_cxx_flag FLAG_TO_CHECK)
  # In order to check for a -Wno-* flag in gcc, you have to check the
  # -W* version instead.  See http://gcc.gnu.org/wiki/FAQ#wnowarning
  string(REGEX REPLACE ^-Wno- -W POSITIVE_FLAG_TO_CHECK ${FLAG_TO_CHECK})
  execute_process(
    COMMAND
    bash -c
    "LC_ALL=POSIX ${CMAKE_CXX_COMPILER} -Werror ${POSITIVE_FLAG_TO_CHECK} \
-x c++ -c - <<< \"\" -o /dev/null"
    RESULT_VARIABLE RESULT
    ERROR_VARIABLE ERROR_FROM_COMPILATION
    OUTPUT_QUIET)
  if(${RESULT} EQUAL 0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG_TO_CHECK}" PARENT_SCOPE)
  endif(${RESULT} EQUAL 0)
endfunction()

# Checks which of the CXX FLAGS_TO_CHECK are supported by the compiler and
# adds them. If adding many flags, this will be much faster than
# calling check_and_add_cxx_flag multiple times.
# - FLAGS_TO_CHECK: a semicolon separated string of CXX flags to try to add
#                   for compilation.
function(check_and_add_cxx_flags FLAGS_TO_CHECK)
  # In order to check for a -Wno-* flag in gcc, you have to check the
  # -W* version instead.  See http://gcc.gnu.org/wiki/FAQ#wnowarning
  set(POSITIVE_FLAGS_TO_CHECK)
  foreach(FLAG_TO_CHECK ${FLAGS_TO_CHECK})
    string(REGEX REPLACE ^-Wno- -W POSITIVE_FLAG_TO_CHECK ${FLAG_TO_CHECK})
    list(APPEND POSITIVE_FLAGS_TO_CHECK ${POSITIVE_FLAG_TO_CHECK})
  endforeach()
  string(REPLACE ";" " "
    POSITIVE_FLAGS_WITH_SPACES "${POSITIVE_FLAGS_TO_CHECK}")
  execute_process(
    COMMAND
    bash -c
    "LC_ALL=POSIX ${CMAKE_CXX_COMPILER} -Werror ${POSITIVE_FLAGS_WITH_SPACES} \
-x c++ -c - <<< \"\" -o /dev/null"
    RESULT_VARIABLE RESULT
    ERROR_VARIABLE ERROR_FROM_COMPILATION
    OUTPUT_QUIET)

  if(${RESULT} EQUAL 0)
    string(REPLACE ";" " " FLAGS_WITH_SPACES "${FLAGS_TO_CHECK}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_WITH_SPACES}" PARENT_SCOPE)
  else(${RESULT} EQUAL 0)
    # Check each flag to see if it was marked as "invalid" in the output
    foreach(FLAG ${POSITIVE_FLAGS_TO_CHECK})
      string(FIND "${ERROR_FROM_COMPILATION}" "'${FLAG}'" FOUND_POS)
      if(${FOUND_POS} EQUAL -1)
        # For some reason:
        # list(FIND ${POSITIVE_FLAGS_TO_CHECK} ${FLAG} INDEX_OF_FLAG)
        # doesn't work with some compilers. This makes no sense but such is
        # life. As a work around we basically implement a find manually.

        # Find the index of the current flag in the POSITIVE_FLAGS_TO_CHECK
        # list. This is the index we use to get the original flag in the
        # FLAGS_TO_CHECK list.
        set(INDEX 0)
        foreach(POS_FLAG ${POSITIVE_FLAGS_TO_CHECK})
          if("${POS_FLAG}" STREQUAL "${FLAG}")
            break()
          endif()
          MATH(EXPR INDEX "${INDEX}+1")
        endforeach()
        set(TARGET_INDEX ${INDEX})
        set(INDEX 0)
        # Get original flag
        set(NEW_FLAG "")
        foreach(ORIGINAL_FLAG ${FLAGS_TO_CHECK})
          if(${INDEX} EQUAL ${TARGET_INDEX})
            set(NEW_FLAG ${ORIGINAL_FLAG})
            break()
          endif()
          MATH(EXPR INDEX "${INDEX}+1")
        endforeach()
        # Add the flag to the list of flags to add.
        set(FLAGS_TO_ADD "${FLAGS_TO_ADD} ${NEW_FLAG}")
      endif(${FOUND_POS} EQUAL -1)
    endforeach(FLAG ${FLAGS_TO_CHECK})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_TO_ADD}" PARENT_SCOPE)
  endif(${RESULT} EQUAL 0)
endfunction()

# Checks if a flag is supported by the linker and adds it if it is
function(check_and_add_cxx_link_flag FLAG_TO_CHECK)
  include(CheckCxxLinkerFlag)
  unset(CXX_LINKER_FLAG_WORKS CACHE)
  set(CMAKE_REQUIRED_QUIET 1)
  check_cxx_linker_flag(${FLAG_TO_CHECK} CXX_LINKER_FLAG_WORKS)
  unset(CMAKE_REQUIRED_QUIET)
  if(CXX_LINKER_FLAG_WORKS)
    set(CMAKE_CXX_LINK_FLAGS
      "${CMAKE_CXX_LINK_FLAGS} ${FLAG_TO_CHECK}" PARENT_SCOPE)
  endif()
endfunction()
