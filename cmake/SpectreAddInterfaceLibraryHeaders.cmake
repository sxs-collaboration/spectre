# Distributed under the MIT License.
# See LICENSE.txt for details.

# Adds the header files to the target.
#
# Usage:
#
#   add_interface_lib_headers(
#     TARGET TARGET_NAME
#     HEADERS
#     A.hpp
#     B.hpp
#     C.hpp
#     )
#
# This function is intended to be used with libraries added using add_library
# or added by CMake's provided find_package (e.g. Boost). The
# add_spectre_library handles adding header files for targets correctly and
# so this function does not need to be used for libraries added with
# add_spectre_library.
function(add_interface_lib_headers)
  cmake_parse_arguments(
    ARG "" "TARGET" "HEADERS"
    ${ARGN})

  if(NOT TARGET ${ARG_TARGET})
    message(FATAL_ERROR
      "Unknown target '${ARG_TARGET}'"
      )
  endif(NOT TARGET ${ARG_TARGET})

  get_target_property(
    TARGET_TYPE
    ${ARG_TARGET}
    TYPE
    )
  if(NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
    message(FATAL_ERROR
      "The target '${ARG_TARGET}' is not an INTERFACE library and so "
      "add_interface_lib_headers should not be used to add header files "
      "to it."
      )
  endif(NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)

  get_property(
    SPECTRE_INTERFACE_LIBRARY_HEADERS
    GLOBAL
    PROPERTY SPECTRE_INTERFACE_LIBRARY_HEADERS
    )

  # Switch to delimiting the headers by ':' because CMake uses ';' to delimit
  # elements of a list.
  string(REPLACE ";" ":" TARGET_HEADERS
    "${ARG_TARGET}=${ARG_HEADERS}")

  # BEGIN ADD EXISTING HEADERS
  # This block checks if the current target already has a list of header files
  # associated with it. If so, then we parse the current headers from the
  # string SPECTRE_INTERFACE_LIBRARY_HEADERS. Once we've parsed out the
  # _PREVIOUS_HEADERS we append them to the list of current headers,
  # (TARGET_HEADERS). Finally, we remove the current target from the interface
  # libaries string (SPECTRE_INTERFACE_LIBRARY_HEADERS) because we add it with
  # the new headers later.
  set(_PREVIOUS_HEADERS "")
  string(FIND "${SPECTRE_INTERFACE_LIBRARY_HEADERS}" "${ARG_TARGET}="
    _POSITION_OF_TARGET)
  if (NOT ${_POSITION_OF_TARGET} EQUAL -1)
    string(SUBSTRING "${SPECTRE_INTERFACE_LIBRARY_HEADERS}"
      ${_POSITION_OF_TARGET} -1 _PREVIOUS_HEADERS)
    string(FIND "${_PREVIOUS_HEADERS}" "="
      POSITION_OF_THIS_EQUALS)
    math(EXPR POSITION_OF_THIS_EQUALS_PLUS_ONE
      "${POSITION_OF_THIS_EQUALS} + 1")
    string(SUBSTRING "${_PREVIOUS_HEADERS}" ${POSITION_OF_THIS_EQUALS_PLUS_ONE}
      -1 _PREVIOUS_HEADERS)
    string(FIND "${_PREVIOUS_HEADERS}" "="
      POSITION_OF_NEXT_EQUALS)
    string(SUBSTRING "${_PREVIOUS_HEADERS}" 0 ${POSITION_OF_NEXT_EQUALS}
      _PREVIOUS_HEADERS)
    string(FIND "${_PREVIOUS_HEADERS}" ";" POSITION_OF_LAST_SEMI REVERSE)
    string(SUBSTRING "${_PREVIOUS_HEADERS}" 0 ${POSITION_OF_LAST_SEMI}
      _PREVIOUS_HEADERS)
    set(TARGET_HEADERS "${TARGET_HEADERS}:${_PREVIOUS_HEADERS}")
    string(REPLACE "${ARG_TARGET}=${_PREVIOUS_HEADERS}" ""
      SPECTRE_INTERFACE_LIBRARY_HEADERS "${SPECTRE_INTERFACE_LIBRARY_HEADERS}")
  endif()

  string(REPLACE ";;" ";"
      SPECTRE_INTERFACE_LIBRARY_HEADERS "${SPECTRE_INTERFACE_LIBRARY_HEADERS}")
  list(APPEND SPECTRE_INTERFACE_LIBRARY_HEADERS ${TARGET_HEADERS})
  # END ADD EXISTING HEADERS

  set_property(
    GLOBAL PROPERTY SPECTRE_INTERFACE_LIBRARY_HEADERS
    ${SPECTRE_INTERFACE_LIBRARY_HEADERS}
    )
endfunction(add_interface_lib_headers)

# Returns a list of all the header files for the target `TARGET`
# by setting the variable with the name `${RESULT_NAME}`
#
# Usage:
#   get_target_headers(MyTarget MY_TARGET_HEADERS)
function(get_target_headers TARGET RESULT_NAME)
  get_target_property(
    TARGET_TYPE
    ${TARGET}
    TYPE
    )
  if(${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
    get_property(
      _SPECTRE_INTERFACE_LIBRARY_HEADERS
      GLOBAL
      PROPERTY SPECTRE_INTERFACE_LIBRARY_HEADERS
      )
    foreach(_LIB ${_SPECTRE_INTERFACE_LIBRARY_HEADERS})
      string(REPLACE "${TARGET}=" "" _LIB_HEADERS ${_LIB})
      if(NOT ${_LIB} STREQUAL ${_LIB_HEADERS})
        string(REPLACE ":" ";" _LIB_HEADERS ${_LIB_HEADERS})
        set(${RESULT_NAME} ${_LIB_HEADERS} PARENT_SCOPE)
        break()
      endif(NOT ${_LIB} STREQUAL ${_LIB_HEADERS})
    endforeach(_LIB ${_SPECTRE_INTERFACE_LIBRARY_HEADERS})

  else(${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
    get_property(
      _HEADER_FILES
      TARGET ${TARGET}
      PROPERTY PUBLIC_HEADER
      )
    set(${RESULT_NAME} ${_HEADER_FILES} PARENT_SCOPE)
  endif(${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
endfunction(get_target_headers TARGET)
