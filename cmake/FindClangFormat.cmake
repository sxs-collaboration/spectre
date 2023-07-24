# Distributed under the MIT License.
# See LICENSE.txt for details.

if(NOT CLANG_FORMAT_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(CLANG_FORMAT_ROOT "")
  set(CLANG_FORMAT_ROOT $ENV{CLANG_FORMAT_ROOT})
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(
      REGEX MATCH "^[0-9]+.[0-9]+" LLVM_VERSION
      "${CMAKE_CXX_COMPILER_VERSION}"
  )
  find_program(
      CLANG_FORMAT_BIN
      NAMES "clang-format-${LLVM_VERSION}" "clang-format"
      HINTS ${CLANG_FORMAT_ROOT}
  )
else()
  find_program(
      CLANG_FORMAT_BIN
      NAMES
      "clang-format-4.0"
      "clang-format-3.9"
      "clang-format-3.8"
      "clang-format"
  )
endif()

if (CLANG_FORMAT_BIN)
  execute_process(COMMAND ${CLANG_FORMAT_BIN} --version
    RESULT_VARIABLE CLANG_FORMAT_VERSION_RESULT
    OUTPUT_VARIABLE CLANG_FORMAT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(CLANG_FORMAT_VERSION_RESULT MATCHES 0)
    string(REGEX REPLACE "clang-format version " ""
      CLANG_FORMAT_VERSION ${CLANG_FORMAT_VERSION})
  endif()
endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  ClangFormat REQUIRED_VARS CLANG_FORMAT_BIN VERSION_VAR CLANG_FORMAT_VERSION
  )

mark_as_advanced(CLANG_FORMAT_VERSION)
