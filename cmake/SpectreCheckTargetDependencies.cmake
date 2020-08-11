# Distributed under the MIT License.
# See LICENSE.txt for details.

include(SpectreAddInterfaceLibraryHeaders)

# Get the list of absolute paths of header files for target `TARGET_NAME`.
#
# Sets the variable `HEADER_FILES` in parent scope to return the result
function(_absolute_header_paths TARGET_NAME)
  get_target_headers(${TARGET_NAME} _HEADER_FILES)
  get_property(
    _INCLUDE_DIR
    TARGET ${TARGET_NAME}
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    )
  list(LENGTH _INCLUDE_DIR _NUMBER_OF_INCLUDE_DIRS)
  if(${_NUMBER_OF_INCLUDE_DIRS} GREATER 1)
    message(FATAL_ERROR
      "Currently _absolute_header_paths only supports a single "
      "INTERFACE_INCLUDE_DIRECTORIES for finding header files. Support for "
      "multiple INTERFACE_INCLUDE_DIRECTORIES can be added if needed.")
  endif(${_NUMBER_OF_INCLUDE_DIRS} GREATER 1)

  unset(_ABS_HEADER_FILES)
  foreach(HEADER ${_HEADER_FILES})
    if(IS_ABSOLUTE ${HEADER})
      list(APPEND _ABS_HEADER_FILES ${HEADER})
    else()
      list(APPEND _ABS_HEADER_FILES "${_INCLUDE_DIR}/${HEADER}")
    endif()
  endforeach(HEADER ${_HEADER_FILES})
  # "Return" HEADER_FILES by setting it in PARENT_SCOPE
  set(HEADER_FILES ${_ABS_HEADER_FILES} PARENT_SCOPE)
endfunction(_absolute_header_paths TARGET_NAME)

# Get the list of absolute paths of source files for target `TARGET_NAME`.
#
# Sets the variable `SOURCE_FILES` in parent scope to return the result
function(_absolute_source_paths TARGET_NAME)
  get_property(
    _SOURCE_FILES
    TARGET ${TARGET_NAME}
    PROPERTY SOURCES
    )
  get_property(
    _LOCATION
    TARGET ${TARGET_NAME}
    PROPERTY FOLDER
    )

  unset(_ABS_SOURCE_FILES)
  foreach(SOURCE ${_SOURCE_FILES})
    if(IS_ABSOLUTE ${SOURCE})
      list(APPEND _ABS_SOURCE_FILES ${SOURCE})
    else()
      list(APPEND _ABS_SOURCE_FILES "${_LOCATION}/${SOURCE}")
    endif()
  endforeach(SOURCE ${_SOURCE_FILES})
  # "Return" SOURCE_FILES by setting it in PARENT_SCOPE
  set(SOURCE_FILES ${_ABS_SOURCE_FILES} PARENT_SCOPE)
endfunction(_absolute_source_paths TARGET_NAME)

# Get the list of SpECTRE includes (i.e. not for 3rd party libraries)
# for the target `TARGET_NAME`.
#
# Sets the variable `SPECTRE_INCLUDES` in parent scope to return the result
function(_extract_spectre_includes FILE_NAME)
  file(READ ${FILE_NAME} FILE_CONTENTS)
  string(REGEX MATCHALL "\n#include \"[^\n]+\""
    _RAW_SPECTRE_INCLUDES ${FILE_CONTENTS})
  unset(SPECTRE_INCLUDES)
  foreach(INCLUDE ${_RAW_SPECTRE_INCLUDES})
    string(REPLACE "\n#include \"" "" INCLUDE "${INCLUDE}")
    string(REPLACE "\"" "" INCLUDE "${INCLUDE}")
    list(APPEND SPECTRE_INCLUDES ${INCLUDE})
  endforeach(INCLUDE ${_RAW_SPECTRE_INCLUDES})
  # "Return" SPECTRE_INCLUDES by setting it in PARENT_SCOPE
  set(SPECTRE_INCLUDES ${SPECTRE_INCLUDES} PARENT_SCOPE)
endfunction(_extract_spectre_includes FILE_NAME)

# Get the list of 3rd party library includes for the target `TARGET_NAME`.
#
# Sets the variable `TPL_INCLUDES` in parent scope to return the result
function(_extract_tpl_includes FILE_NAME)
  file(READ ${FILE_NAME} FILE_CONTENTS)
  string(REGEX MATCHALL "\n#include <[^\n]+>"
    _RAW_TPL_INCLUDES ${FILE_CONTENTS})
  unset(TPL_INCLUDES)
  foreach(INCLUDE ${_RAW_TPL_INCLUDES})
    string(REPLACE "\n#include <" "" INCLUDE "${INCLUDE}")
    string(REPLACE ">" "" INCLUDE "${INCLUDE}")
    list(APPEND TPL_INCLUDES ${INCLUDE})
  endforeach(INCLUDE ${_RAW_TPL_INCLUDES})
  # "Return" TPL_INCLUDES by setting it in PARENT_SCOPE
  set(TPL_INCLUDES ${TPL_INCLUDES} PARENT_SCOPE)
endfunction(_extract_tpl_includes FILE_NAME)

# Make the path of `HEADER_FILE` which belongs to the target `TARGET_NAME`
# an absolute path.
#
# Sets the variable `ABS_HEADER_FILE` in parent scope to return the result
function(_make_header_path_absolute TARGET_NAME HEADER_FILE)
  get_property(
    _INCLUDE_DIR
    TARGET ${TARGET_NAME}
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    )
  if(IS_ABSOLUTE ${HEADER_FILE})
    set(_ABS_HEADER_FILE ${HEADER_FILE})
  else()
    set(_ABS_HEADER_FILE "${_INCLUDE_DIR}/${HEADER_FILE}")
  endif()
  # "Return" ABS_HEADER_FILE by setting it in PARENT_SCOPE
  set(ABS_HEADER_FILE ${_ABS_HEADER_FILE} PARENT_SCOPE)
endfunction(_make_header_path_absolute TARGET_NAME HEADER_FILE)

# Gets the dependencies for a list of includes of one of the source or
# header files of the target `TARGET_NAME`
#
# The `_get_deps_for_includes` function builds a list of all dependencies
# by finding the target that provides each header file in `INCLUDES_OF_FILE`.
# `INCLUDES_OF_FILE` must be a list of includes from a file (either header
# or source file) of the target `TARGET_NAME`. If `RECURSE_THROUGH_INCLUDES`
# is `TRUE` then for each header file in `INCLUDES_OF_FILE` that is also an
# include of the target `TARGET_NAME` then the header file will also have its
# dependencies searched and added to the list of dependencies.
#
# Example usage:
#
#   _get_deps_for_includes(
#     TARGET_C
#     "TARGET_A;TARGET_B;TARGET_C;TARGET_D"
#     "cmath;utility;vector"
#     FALSE
#     )
#
# Arguments:
#
# TARGET_NAME: the target for which to check the link libraries/dependencies
# LIST_OF_ALL_TARGETS: a list of all the targets that might be dependencies
#                      of the target ${TARGET_NAME}
# INCLUDES_OF_FILE:
#     a list of header files for which to get the dependencies.
# RECURSE_THROUGH_INCLUDES:
#     if TRUE then any include in any of ${INCLUDES_OF_FILE}
#     that is also an include of the target ${TARGET_NAME} has its
#     dependencies added.
#
# Since the recursion will cause an infinite loop if there are cyclic includes
# we limit the recursion depth to 20, at which point an error message is
# displayed that there may be cyclic dependencies
#
# Returns by setting `TARGET_DEPS` in parent scope
function(_get_deps_for_includes TARGET_NAME LIST_OF_ALL_TARGETS
    INCLUDES_OF_FILE RECURSE_THROUGH_INCLUDES)
  # Implement guard against infinite recursion when there are cyclic
  # includes in a library.
  get_property(
    GET_DEPS_FOR_SPECTRE_INCLUDES_PATH
    GLOBAL PROPERTY GET_DEPS_FOR_SPECTRE_INCLUDES_PATH
    )
  list(LENGTH GET_DEPS_FOR_SPECTRE_INCLUDES_PATH RECURSE_DEPTH)
  if(${RECURSE_DEPTH} GREATER 20)
    message(FATAL_ERROR
      " Found what appears to be a cyclic loop in the header files."
      " The header path is: ${GET_DEPS_FOR_SPECTRE_INCLUDES_PATH}"
      )
  endif(${RECURSE_DEPTH} GREATER 20)

  get_target_headers(${TARGET_NAME} TARGET_HEADER_FILES)

  unset(_TARGET_DEPS)
  # Now loop over all spectre targets to see if any of them
  # expose these includes
  foreach(CURRENT_INCLUDE ${INCLUDES_OF_FILE})
    # If we are checking header files and the header file is part of
    # this target, possibly recurse the header files, and skipping search
    # through targets for which one provides the header file (we know it's us).
    if(${CURRENT_INCLUDE} IN_LIST TARGET_HEADER_FILES)
      if(RECURSE_THROUGH_INCLUDES)
        # Sets ABS_HEADER_FILE
        _make_header_path_absolute(${TARGET_NAME} ${CURRENT_INCLUDE})
        # Sets SPECTRE_INCLUDES
        _extract_spectre_includes(${ABS_HEADER_FILE})
        set_property(
          GLOBAL APPEND PROPERTY GET_DEPS_FOR_CURRENT_INCLUDES_PATH
          ${CURRENT_INCLUDE}
          )
        # Sets TARGET_DEPS
        _get_deps_for_includes(${TARGET_NAME} "${LIST_OF_ALL_TARGETS}"
          "${CURRENT_INCLUDES}" ${RECURSE_THROUGH_INCLUDES})
        # "Pop" last element by setting the property to what it was at the
        # start of the function call.
        set_property(
          GLOBAL PROPERTY GET_DEPS_FOR_CURRENT_INCLUDES_PATH
          ${GET_DEPS_FOR_CURRENT_INCLUDES_PATH}
          )
        # Get the dependencies of the header file, since these are all
        # also PRIVATE dependencies of the target.
        foreach(_TARGET ${TARGET_DEPS})
          if(NOT ${_TARGET} IN_LIST _TARGET_DEPS)
            list(APPEND _TARGET_DEPS ${_TARGET})
          endif(NOT ${_TARGET} IN_LIST _TARGET_DEPS)
        endforeach(_TARGET ${TARGET_DEPS})
        continue()
      else(RECURSE_THROUGH_INCLUDES)
        continue()
      endif(RECURSE_THROUGH_INCLUDES)
    endif(${CURRENT_INCLUDE} IN_LIST TARGET_HEADER_FILES)

    unset(_TARGET_WITH_INCLUDE)
    foreach(DEP_TARGET ${LIST_OF_ALL_TARGETS})
      if(${DEP_TARGET} STREQUAL ${TARGET_NAME})
        continue()
      endif(${DEP_TARGET} STREQUAL ${TARGET_NAME})

      # Get the header files of DEP_TARGET and see if any of the
      # includes are part of DEP_TARGET.
      get_target_headers(${DEP_TARGET} _DEP_TARGET_HEADERS)
      if(${CURRENT_INCLUDE} IN_LIST _DEP_TARGET_HEADERS)
        set(_TARGET_WITH_INCLUDE ${DEP_TARGET})
        break()
      endif(${CURRENT_INCLUDE} IN_LIST _DEP_TARGET_HEADERS)
    endforeach(DEP_TARGET ${LIST_OF_ALL_TARGETS})
    if(NOT _TARGET_WITH_INCLUDE)
      message(FATAL_ERROR
        "No known targets supply the file ${CURRENT_INCLUDE} included in "
        "file ${HEADER_OR_SOURCE} of target ${TARGET_NAME}."
        )
    endif(NOT _TARGET_WITH_INCLUDE)

    if(NOT ${_TARGET_WITH_INCLUDE} IN_LIST _TARGET_DEPS)
      list(APPEND _TARGET_DEPS ${_TARGET_WITH_INCLUDE})
    endif(NOT ${_TARGET_WITH_INCLUDE} IN_LIST _TARGET_DEPS)
  endforeach(CURRENT_INCLUDE ${INCLUDES_OF_FILE})

  # "Return" TARGET_DEPS by setting it in PARENT_SCOPE
  set(TARGET_DEPS ${_TARGET_DEPS} PARENT_SCOPE)
endfunction(_get_deps_for_includes TARGET_NAME LIST_OF_ALL_TARGETS
  INCLUDES_OF_FILE RECURSE_THROUGH_INCLUDES)

# Returns the dependencies for the target
#
# Arguments
#
# TARGET_NAME:
#     the target for which to check the link libraries/dependencies
# LIST_OF_ALL_TARGETS:
#     a list of all the targets that might be dependencies
#     of the target ${TARGET_NAME}
# LIST_OF_FILES:
#     a list of the files for which to get the dependencies. Can be
#     header/tpp or source files.
# RECURSE_INCLUDES:
#     if TRUE then the dependencies of the target's included header files
#     will also be added. This is used when source files are passed because
#     the dependencies of the header files must be marked as PUBLIC
#
# Returns by setting `TARGET_DEPS` in parent scope
function(_get_deps_for_target TARGET_NAME LIST_OF_ALL_TARGETS
    LIST_OF_FILES RECURSE_INCLUDES)
  unset(_TARGET_DEPS)
  foreach(HEADER_OR_SOURCE ${LIST_OF_FILES})
    if(IS_ABSOLUTE "${HEADER_OR_SOURCE}")
      # Sets SPECTRE_INCLUDES
      _extract_spectre_includes("${HEADER_OR_SOURCE}")
    else()
      message(FATAL_ERROR
        "All paths of files passed to _get_deps_for_target must be absolute "
        "but received ${HEADER_OR_SOURCE}.")
    endif()

    # Sets TARGET_DEPS
    _get_deps_for_includes(${TARGET_NAME} "${LIST_OF_ALL_TARGETS}"
      "${SPECTRE_INCLUDES}" ${RECURSE_INCLUDES})
    foreach(_TARGET ${TARGET_DEPS})
      if(NOT ${_TARGET} IN_LIST _TARGET_DEPS)
        list(APPEND _TARGET_DEPS ${_TARGET})
      endif(NOT ${_TARGET} IN_LIST _TARGET_DEPS)
    endforeach(_TARGET ${TARGET_DEPS})

    # Third party library includes

    # Sets TPL_INCLUDES
    _extract_tpl_includes("${HEADER_OR_SOURCE}")
    # Sets TARGET_DEPS
    _get_deps_for_includes(${TARGET_NAME} "${LIST_OF_ALL_TARGETS}"
      "${TPL_INCLUDES}" FALSE)
    foreach(_TARGET ${TARGET_DEPS})
      if(NOT ${_TARGET} IN_LIST _TARGET_DEPS)
        list(APPEND _TARGET_DEPS ${_TARGET})
      endif(NOT ${_TARGET} IN_LIST _TARGET_DEPS)
    endforeach(_TARGET ${TARGET_DEPS})

  endforeach(HEADER_OR_SOURCE ${LIST_OF_FILES})
  # Alphabetize the list of dependencies
  list(SORT _TARGET_DEPS)
  # "Return" the target dependencies
  set(TARGET_DEPS ${_TARGET_DEPS} PARENT_SCOPE)
endfunction(_get_deps_for_target TARGET_NAME LIST_OF_FILES LIST_OF_ALL_TARGETS)

# Add the libraries for the `target_link_libraries` command
# for a target.
#
# Arguments
#
# TARGET_DEPS:
#     dependencies of the target
# TARGET_DEPS_TYPE:
#     the type of dependencies, INTERFACE/PRIVATE/PUBLIC
# TARGET_LINK_LIBRARIES_COMMAND:
#     the target_link_libraries command so far
#
# Returns by setting `TARGET_LINK_LIBS_COMMAND` in parent scope
function(_add_targets_to_link_libs TARGET_DEPS TARGET_DEPS_TYPE
    TARGET_LINK_LIBS_COMMAND)
  list(LENGTH TARGET_DEPS _NUM_TARGET_DEPS)
  set(_TARGET_LINK_LIBS_COMMAND
    "${TARGET_LINK_LIBS_COMMAND}"
    )
  if(${_NUM_TARGET_DEPS} GREATER 0)
    set(
      _TARGET_LINK_LIBS_COMMAND
      "${_TARGET_LINK_LIBS_COMMAND}\n"
      "   ${TARGET_DEPS_TYPE}"
      )
    foreach(_DEP ${TARGET_DEPS})
      set(
        _TARGET_LINK_LIBS_COMMAND
        "${_TARGET_LINK_LIBS_COMMAND}\n"
        "   ${_DEP}"
        )
    endforeach(_DEP ${_TARGET_DEPS})
    set(_TARGET_LINK_LIBS_COMMAND
      "${_TARGET_LINK_LIBS_COMMAND}\n "
      )
    string(REPLACE ";" "" _TARGET_LINK_LIBS_COMMAND
      ${_TARGET_LINK_LIBS_COMMAND})
  endif(${_NUM_TARGET_DEPS} GREATER 0)
  # "Return" the target dependencies
  set(TARGET_LINK_LIBS_COMMAND ${_TARGET_LINK_LIBS_COMMAND} PARENT_SCOPE)
endfunction(_add_targets_to_link_libs TARGET_DEPS TARGET_DEPS_TYPE
  TARGET_LINK_LIBS_COMMAND)

# Queries the INTERFACE_LINK_LIBRARIES of TARGET. This is done recursively.
function(_get_interface_link_libraries TARGET)
  get_target_property(_INTERFACE_LIBS ${TARGET} INTERFACE_LINK_LIBRARIES)
  set(LINK_LIBS_FOR_TARGET "")
  foreach(_INTERFACE_LIB ${_INTERFACE_LIBS})
    if (_INTERFACE_LIB)
      list(APPEND LINK_LIBS_FOR_TARGET ${_INTERFACE_LIB})
      if (TARGET ${_INTERFACE_LIB})
        set(OUTPUT_LIST "")
        _get_interface_link_libraries(${_INTERFACE_LIB})
        list(APPEND LINK_LIBS_FOR_TARGET ${OUTPUT_LIST})
      endif()
    endif()
  endforeach()
  # "Return" dependencies to calling function
  set(OUTPUT_LIST ${LINK_LIBS_FOR_TARGET} PARENT_SCOPE)
endfunction(_get_interface_link_libraries OUTPUT_LIST TARGET)

# Checks the dependencies for the target ${TARGET_NAME} and if they are
# incorrect produces an error message with the correct dependencies.
#
# TARGET_NAME:
#     the target for which to check the link libraries/dependencies
# TARGET_LINK_LIBS_COMMAND:
#     the target_link_libraries CMake command so far
# TARGET_INTERFACE_DEPS:
#     the INTERFACE link libraries for the target
# TARGET_PRIVATE_DEPS:
#     the PRIVATE link libraries for the target
# TARGET_PUBLIC_DEPS:
#     the PUBLIC link libraries for the target
# LIST_OF_ALLOWED_EXTRA_TARGETS:
#     a list of allowed extra dependencies. An example of an extra dependency
#     is the SpectreFlags target, which supplies no header files and there is
#     not strictly a dependency of ${TARGET_NAME} but can be specified as a
#     dependency in order to add compiler flags or definitions to the target.
# ERROR_ON_FAILURE:
#     if specified, then CMake will produce an error if the dependencies are
#     incorrect. Otherwise the variable TARGET_DEPENDENCIES_ERROR_MESSAGE
#     is set in the parent scope
function(_check_and_print_dependencies
    TARGET_NAME TARGET_LINK_LIBS_COMMAND
    TARGET_INTERFACE_DEPS TARGET_PRIVATE_DEPS TARGET_PUBLIC_DEPS
    LIST_OF_ALLOWED_EXTRA_TARGETS)
  cmake_parse_arguments(
    ARG "ERROR_ON_FAILURE" "" "" ${ARGN})

  # We add the dependencies of allowed targets as additional allowed targets.
  # This avoids the situation where target A is allowed, but A depends on B,
  # and therefore every target has a dependency on B. The helper function
  # ensures the dependencies are added recursively.
  set(WORKING_LIST ${LIST_OF_ALLOWED_EXTRA_TARGETS})
  foreach(_ALLOWED_EXTRA_TARGET ${LIST_OF_ALLOWED_EXTRA_TARGETS})
    set(OUTPUT_LIST "")
    _get_interface_link_libraries(${_ALLOWED_EXTRA_TARGET})
    list(APPEND WORKING_LIST ${OUTPUT_LIST})
  endforeach()
  list(REMOVE_DUPLICATES WORKING_LIST)
  set(LIST_OF_ALLOWED_EXTRA_TARGETS ${WORKING_LIST})

  get_property(
    _INTERFACE_LIBS
    TARGET ${TARGET_NAME}
    PROPERTY INTERFACE_LINK_LIBRARIES
    )
  # The property INTERFACE_LINK_LIBRARIES contains generator expressions for
  # private link-time dependencies. We will be handling these private
  # dependencies explicitly below, so here we want to remove them from the list
  # of interface dependencies. The generators have format "$<LINK_ONLY:dep>"
  list(FILTER _INTERFACE_LIBS EXCLUDE REGEX "\\$<LINK_ONLY:.+>")

  unset(_MISSING_INTERFACE_LIBS)
  foreach(_INTERFACE_DEP ${TARGET_INTERFACE_DEPS})
    if(NOT ${_INTERFACE_DEP} IN_LIST _INTERFACE_LIBS)
      list(APPEND _MISSING_INTERFACE_LIBS ${_INTERFACE_DEP})
    endif(NOT ${_INTERFACE_DEP} IN_LIST _INTERFACE_LIBS)
  endforeach(_INTERFACE_DEP ${TARGET_INTERFACE_DEPS})

  get_target_property(
    TARGET_TYPE
    ${TARGET_NAME}
    TYPE
    )
  set(_NUM_MISSING_PRIVATE_LIBS 0)
  set(_NUM_MISSING_PUBLIC_LIBS 0)
  unset(_EXTRA_INTERFACE_LIBS)
  unset(_EXTRA_PRIVATE_LIBS)
  unset(_EXTRA_PUBLIC_LIBS)
  if(NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
    get_property(
      _PRIVATE_LIBS
      TARGET ${TARGET_NAME}
      PROPERTY LINK_LIBRARIES
      )

    # Check to see if any interface-only dependencies are also
    # PRIVATE/PUBLIC
    foreach(_INTERFACE_DEP ${TARGET_INTERFACE_DEPS})
      if(${_INTERFACE_DEP} IN_LIST _PRIVATE_LIBS)
        list(APPEND _MISSING_INTERFACE_LIBS ${_INTERFACE_DEP})
      endif(${_INTERFACE_DEP} IN_LIST _PRIVATE_LIBS)
    endforeach(_INTERFACE_DEP ${TARGET_INTERFACE_DEPS})

    unset(_MISSING_PRIVATE_LIBS)
    foreach(_PRIVATE_DEP ${TARGET_PRIVATE_DEPS})
      if(NOT ${_PRIVATE_DEP} IN_LIST _PRIVATE_LIBS)
        list(APPEND _MISSING_PRIVATE_LIBS ${_PRIVATE_DEP})
      endif(NOT ${_PRIVATE_DEP} IN_LIST _PRIVATE_LIBS)
    endforeach(_PRIVATE_DEP ${TARGET_PRIVATE_DEPS})
    list(LENGTH _MISSING_PRIVATE_LIBS _NUM_MISSING_PRIVATE_LIBS)

    unset(_MISSING_PUBLIC_LIBS)
    foreach(_PUBLIC_DEP ${TARGET_PUBLIC_DEPS})
      if(NOT ${_PUBLIC_DEP} IN_LIST _PRIVATE_LIBS
          OR NOT ${_PUBLIC_DEP} IN_LIST _INTERFACE_LIBS)
        list(APPEND _MISSING_PUBLIC_LIBS ${_PUBLIC_DEP})
      endif(NOT ${_PUBLIC_DEP} IN_LIST _PRIVATE_LIBS
        OR NOT ${_PUBLIC_DEP} IN_LIST _INTERFACE_LIBS)
    endforeach(_PUBLIC_DEP ${TARGET_PUBLIC_DEPS})
    list(LENGTH _MISSING_PUBLIC_LIBS _NUM_MISSING_PUBLIC_LIBS)

    # Now check for extra target link libraries that aren't needed.
    unset(_PUBLIC_LIBS)
    foreach(_INTERFACE_LIB ${_INTERFACE_LIBS})
      if(${_INTERFACE_LIB} IN_LIST _PRIVATE_LIBS)
        list(APPEND _PUBLIC_LIBS ${_INTERFACE_LIB})
      endif(${_INTERFACE_LIB} IN_LIST _PRIVATE_LIBS)
    endforeach(_INTERFACE_LIB ${_INTERFACE_LIBS})

    foreach(_PUBLIC_LIB ${_PUBLIC_LIBS})
      list(REMOVE_ITEM _INTERFACE_LIBS ${_PUBLIC_LIB})
      list(REMOVE_ITEM _PRIVATE_LIBS ${_PUBLIC_LIB})
      if(NOT ${_PUBLIC_LIB} IN_LIST TARGET_PUBLIC_DEPS AND
          NOT ${_PUBLIC_LIB} IN_LIST LIST_OF_ALLOWED_EXTRA_TARGETS)
        list(APPEND _EXTRA_PUBLIC_LIBS ${_PUBLIC_LIB})
      endif(NOT ${_PUBLIC_LIB} IN_LIST TARGET_PUBLIC_DEPS AND
          NOT ${_PUBLIC_LIB} IN_LIST LIST_OF_ALLOWED_EXTRA_TARGETS)
    endforeach(_PUBLIC_LIB ${_PUBLIC_LIBS})

    foreach(_PRIVATE_LIB ${_PRIVATE_LIBS})
      if(NOT ${_PRIVATE_LIB} IN_LIST TARGET_PRIVATE_DEPS AND
          NOT ${_PRIVATE_LIB} IN_LIST LIST_OF_ALLOWED_EXTRA_TARGETS)
        list(APPEND _EXTRA_PRIVATE_LIBS ${_PRIVATE_LIB})
      endif(NOT ${_PRIVATE_LIB} IN_LIST TARGET_PRIVATE_DEPS AND
          NOT ${_PRIVATE_LIB} IN_LIST LIST_OF_ALLOWED_EXTRA_TARGETS)
    endforeach(_PRIVATE_LIB ${_PRIVATE_LIBS})

    foreach(_INTERFACE_LIB ${_INTERFACE_LIBS})
      if(NOT ${_INTERFACE_LIB} IN_LIST TARGET_INTERFACE_DEPS AND
          NOT ${_INTERFACE_LIB} IN_LIST LIST_OF_ALLOWED_EXTRA_TARGETS)
        list(APPEND _EXTRA_INTERFACE_LIBS ${_INTERFACE_LIB})
      endif(NOT ${_INTERFACE_LIB} IN_LIST TARGET_INTERFACE_DEPS AND
          NOT ${_INTERFACE_LIB} IN_LIST LIST_OF_ALLOWED_EXTRA_TARGETS)
    endforeach(_INTERFACE_LIB ${_INTERFACE_LIBS})
  else(NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
    foreach(_INTERFACE_DEP ${_INTERFACE_LIBS})
      if(NOT ${_INTERFACE_DEP} IN_LIST TARGET_INTERFACE_DEPS AND
          NOT ${_INTERFACE_DEP} IN_LIST LIST_OF_ALLOWED_EXTRA_TARGETS)
        list(APPEND _EXTRA_INTERFACE_LIBS ${_INTERFACE_DEP})
      endif()
    endforeach(_INTERFACE_DEP ${_INTERFACE_LIBS})
  endif(NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)

  list(LENGTH _MISSING_INTERFACE_LIBS _NUM_MISSING_INTERFACE_LIBS)

  list(LENGTH _EXTRA_INTERFACE_LIBS _NUM_EXTRA_INTERFACE_LIBS)
  list(LENGTH _EXTRA_PRIVATE_LIBS _NUM_EXTRA_PRIVATE_LIBS)
  list(LENGTH _EXTRA_PUBLIC_LIBS _NUM_EXTRA_PUBLIC_LIBS)

  set(_MISSING_LIBS FALSE)
  set(_ERROR_MSG " Missing dependencies for target ${TARGET_NAME}")
  if(${_NUM_MISSING_INTERFACE_LIBS} GREATER 0)
    set(_ERROR_MSG
      "${_ERROR_MSG}\n Missing interface libs:\n   ${_MISSING_INTERFACE_LIBS}")
    set(_MISSING_LIBS TRUE)
  endif(${_NUM_MISSING_INTERFACE_LIBS} GREATER 0)
  if(${_NUM_MISSING_PRIVATE_LIBS} GREATER 0)
    set(_ERROR_MSG
      "${_ERROR_MSG}\n Missing private libs:\n   ${_MISSING_PRIVATE_LIBS}")
    set(_MISSING_LIBS TRUE)
  endif(${_NUM_MISSING_PRIVATE_LIBS} GREATER 0)
  if(${_NUM_MISSING_PUBLIC_LIBS} GREATER 0)
    set(_ERROR_MSG
      "${_ERROR_MSG}\n Missing public libs:\n   ${_MISSING_PUBLIC_LIBS}")
    set(_MISSING_LIBS TRUE)
  endif(${_NUM_MISSING_PUBLIC_LIBS} GREATER 0)

  if(${_NUM_EXTRA_INTERFACE_LIBS} GREATER 0 OR
      ${_NUM_EXTRA_PRIVATE_LIBS} GREATER 0 OR
      ${_NUM_EXTRA_PUBLIC_LIBS} GREATER 0)
    set(_ERROR_MSG " Extra dependencies for target ${TARGET_NAME}")
  endif()
  if(${_NUM_EXTRA_INTERFACE_LIBS} GREATER 0)
    set(_ERROR_MSG
      "${_ERROR_MSG}\n Extra interface libs:\n   ${_EXTRA_INTERFACE_LIBS}")
    set(_MISSING_LIBS TRUE)
  endif(${_NUM_EXTRA_INTERFACE_LIBS} GREATER 0)
  if(${_NUM_EXTRA_PRIVATE_LIBS} GREATER 0)
    set(_ERROR_MSG
      "${_ERROR_MSG}\n Extra private libs:\n   ${_EXTRA_PRIVATE_LIBS}")
    set(_MISSING_LIBS TRUE)
  endif(${_NUM_EXTRA_PRIVATE_LIBS} GREATER 0)
  if(${_NUM_EXTRA_PUBLIC_LIBS} GREATER 0)
    set(_ERROR_MSG
      "${_ERROR_MSG}\n Extra public libs:\n   ${_EXTRA_PUBLIC_LIBS}")
    set(_MISSING_LIBS TRUE)
  endif(${_NUM_EXTRA_PUBLIC_LIBS} GREATER 0)

  if(NOT ARG_ERROR_ON_FAILURE)
    set(
      TARGET_DEPS_ERROR_MESSAGE
      ""
      PARENT_SCOPE
      )
  endif(NOT ARG_ERROR_ON_FAILURE)

  if(_MISSING_LIBS)
    set(TARGET_LINK_LIBS_COMMAND
      " target_link_libraries(\n"
      "   ${TARGET_NAME}"
      )
    _add_targets_to_link_libs(
      "${TARGET_INTERFACE_DEPS}" "INTERFACE"
      "${TARGET_LINK_LIBS_COMMAND}")
    _add_targets_to_link_libs(
      "${TARGET_PRIVATE_DEPS}" "PRIVATE"
      "${TARGET_LINK_LIBS_COMMAND}")
    _add_targets_to_link_libs(
      "${TARGET_PUBLIC_DEPS}" "PUBLIC"
      "${TARGET_LINK_LIBS_COMMAND}")
    set(TARGET_LINK_LIBS_COMMAND
      "${TARGET_LINK_LIBS_COMMAND}  )"
      )

    if(${ARG_ERROR_ON_FAILURE})
      message(FATAL_ERROR
        "${_ERROR_MSG}\n \n"
        " Correct target_link_libraries command is:\n"
        "${TARGET_LINK_LIBS_COMMAND}"
        )
    else(${ARG_ERROR_ON_FAILURE})
      set(
        TARGET_DEPS_ERROR_MESSAGE
        "${_ERROR_MSG}\n \n\
 Correct target_link_libraries command is:\n\
${TARGET_LINK_LIBS_COMMAND}"
        PARENT_SCOPE
        )
    endif(${ARG_ERROR_ON_FAILURE})
  endif(_MISSING_LIBS)
endfunction(_check_and_print_dependencies
  TARGET_NAME TARGET_LINK_LIBS_COMMAND
  TARGET_INTERFACE_DEPS TARGET_PRIVATE_DEPS TARGET_PUBLIC_DEPS
  LIST_OF_ALLOWED_EXTRA_TARGETS)

# Check that the link libraries of the target are correct and if not,
# provide an error with the correct target link libraries.
#
# Example usage:
#
#    check_target_dependencies(
#      TARGET ${TARGET_TO_CHECK}
#      ALL_TARGETS
#      ${SPECTRE_TPLS}
#      ${SPECTRE_LIBS}
#      ALLOWED_EXTRA_TARGETS
#      SpectreFlags
#      ERROR_ON_FAILURE
#      )
#
# Arguments:
#
# TARGET:
#     the target for which to check the link libraries/dependencies
# ALL_TARGETS:
#     a list of all the targets that might be dependencies of the target
#     ${TARGET}
# ALLOWED_EXTRA_TARGETS:
#     a list of allowed extra dependencies. An example of an extra dependency
#     is the SpectreFlags target, which supplies no header files and therefore
#     is not strictly a dependency of ${TARGET} but can be specified as
#     dependency in order to add compiler flags or definitions to the target.
# ERROR_ON_FAILURE:
#     if specified, then CMake will produce an error if the dependencies are
#     incorrect. Otherwise the variable TARGET_DEPENDENCIES_ERROR_MESSAGE
#     is set in the parent scope
function(check_target_dependencies)
  cmake_parse_arguments(
    ARG
    "ERROR_ON_FAILURE"
    "TARGET"
    "ALL_TARGETS;ALLOWED_EXTRA_TARGETS" ${ARGN})

  # Sets HEADER_FILES
  _absolute_header_paths(${ARG_TARGET})

  # Sets TARGET_DEPS
  _get_deps_for_target(${ARG_TARGET}
    "${ARG_ALL_TARGETS}" "${HEADER_FILES}" FALSE)
  set(_TARGET_INTERFACE_DEPS ${TARGET_DEPS})
  list(REMOVE_ITEM _TARGET_INTERFACE_DEPS Stl)

  get_target_property(
    TARGET_TYPE
    ${ARG_TARGET}
    TYPE
    )

  unset(_TARGET_PRIVATE_DEPS)
  unset(_TARGET_PUBLIC_DEPS)
  if(NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
    # Sets SOURCE_FILES
    _absolute_source_paths(${ARG_TARGET})

    # Sets TARGET_DEPS
    _get_deps_for_target(${ARG_TARGET}
      "${ARG_ALL_TARGETS}" "${SOURCE_FILES}" TRUE)
    set(_TARGET_PRIVATE_DEPS ${TARGET_DEPS})
    # We only use the Stl target to be able to keep track of all header files
    # and to be able to claim with certainly that an extra header file is
    # present. Since there are no explicit includes or link dependencies that
    # need to be added, we don't want the extra noise of explicitly depending
    # on Stl everywhere in the code and so we remove it from the list.
    list(REMOVE_ITEM _TARGET_PRIVATE_DEPS Stl)

    # Now need to figure out what the PUBLIC libraries are
    unset(_TARGET_PUBLIC_DEPS)
    foreach(_PRIVATE_DEP ${_TARGET_PRIVATE_DEPS})
      if(${_PRIVATE_DEP} IN_LIST _TARGET_INTERFACE_DEPS)
        list(APPEND _TARGET_PUBLIC_DEPS ${_PRIVATE_DEP})
      endif(${_PRIVATE_DEP} IN_LIST _TARGET_INTERFACE_DEPS)
    endforeach(_PRIVATE_DEP ${_TARGET_PRIVATE_DEPS})
    # Remove PRIVATE deps from INTERFACE and PUBLIC lists
    foreach(_PUBLIC_DEP ${_TARGET_PUBLIC_DEPS})
      list(REMOVE_ITEM _TARGET_PRIVATE_DEPS ${_PUBLIC_DEP})
      list(REMOVE_ITEM _TARGET_INTERFACE_DEPS ${_PUBLIC_DEP})
    endforeach(_PUBLIC_DEP ${_TARGET_PUBLIC_DEPS})
  endif(NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)

  set(ERROR_ON_FAILURE "")
  if(ARG_ERROR_ON_FAILURE)
    set(ERROR_ON_FAILURE ERROR_ON_FAILURE)
  endif(ARG_ERROR_ON_FAILURE)
  _check_and_print_dependencies(
    ${ARG_TARGET}
    "${TARGET_LINK_LIBS_COMMAND}"
    "${_TARGET_INTERFACE_DEPS}"
    "${_TARGET_PRIVATE_DEPS}"
    "${_TARGET_PUBLIC_DEPS}"
    "${ARG_ALLOWED_EXTRA_TARGETS}"
    ${ERROR_ON_FAILURE}
    )
  if(NOT ARG_ERROR_ON_FAILURE)
    set(
      TARGET_DEPS_ERROR_MESSAGE
      "${TARGET_DEPS_ERROR_MESSAGE}"
      PARENT_SCOPE
      )
  endif(NOT ARG_ERROR_ON_FAILURE)
endfunction(check_target_dependencies)
