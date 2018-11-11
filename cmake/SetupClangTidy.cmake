# Distributed under the MIT License.
# See LICENSE.txt for details.

# Remove the checks because:
# -cppcoreguidelines-no-malloc: we want to use malloc with unique_ptr because
#                               not initializing the memory is faster.
# -llvm-header-guard: We use pragma once instead of include guards
# -google-runtime-int: specifying int32_t and int64_t instead of just int
# -readability-else-after-return: style choice, discussed in issue #145
# -misc-noexcept-move-constructor: false positives
# -misc-unconventional-assign-operator: false positives
# -cppcoreguidelines-c-copy-assignment-signature: false positives
# -cert-err58-cpp: many static variables we use do not throw, and if they do
#                  we want to terminate anyway
# -google-default-arguments: defaulting virtual functions in CoordinateMap
#
# Notes:
# misc-move-const-arg: we keep this check because even though this gives
#                      a lot of annoying warnings about moving trivially
#                      copyable types, it warns about moving const objects,
#                      which can have severe performance impacts.
set(CLANG_TIDY_IGNORE_CHECKS "*,-cppcoreguidelines-no-malloc,-llvm-header-guard,-google-runtime-int,-readability-else-after-return,-misc-noexcept-move-constructor,-misc-unconventional-assign-operator,-cppcoreguidelines-c-copy-assignment-signature,-modernize-raw-string-literal,-hicpp-noexcept-move,-hicpp-no-assembler,-android-*,-cert-err58-cpp,-google-default-arguments,-fuchsia-*,-performance-noexcept-move-constructor")

if(NOT CMAKE_CXX_CLANG_TIDY AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(
      REGEX MATCH "^[0-9]+.[0-9]+" LLVM_VERSION
      "${CMAKE_CXX_COMPILER_VERSION}"
  )
  find_program(
      CLANG_TIDY_BIN
      NAMES "clang-tidy-${LLVM_VERSION}" "clang-tidy"
      HINTS ${COMPILER_PATH}
      )
elseif(CMAKE_CXX_CLANG_TIDY)
  set(CLANG_TIDY_BIN "${CMAKE_CXX_CLANG_TIDY}")
endif()

if (CLANG_TIDY_BIN)
  message(STATUS "clang-tidy: ${CLANG_TIDY_BIN}")
  set(MODULES_TO_DEPEND_ON
    module_RunTests
    module_ConstGlobalCache
    module_Main
    module_Test_ConstGlobalCache
    )
  if (TARGET pch)
    list(APPEND MODULES_TO_DEPEND_ON pch)
  endif()
  configure_file(
    ${CMAKE_SOURCE_DIR}/tools/ClangTidyAll.sh
    ${CMAKE_BINARY_DIR}/ClangTidyAll.sh
    @ONLY IMMEDIATE
    )
  add_custom_target(
      clang-tidy
      COMMAND
      ${CLANG_TIDY_BIN}
      -header-filter=${CMAKE_SOURCE_DIR}
      -checks=${CLANG_TIDY_IGNORE_CHECKS}
      -p ${CMAKE_BINARY_DIR}
      \${FILE}
  )
  add_dependencies(
    clang-tidy
    ${MODULES_TO_DEPEND_ON}
    )
  set_target_properties(
      clang-tidy
      PROPERTIES EXCLUDE_FROM_ALL TRUE
  )
  add_custom_target(
      clang-tidy-all
      COMMAND ${CMAKE_BINARY_DIR}/ClangTidyAll.sh
  )
  set_target_properties(
      clang-tidy-all
      PROPERTIES EXCLUDE_FROM_ALL TRUE
  )
  add_dependencies(
    clang-tidy-all
    ${MODULES_TO_DEPEND_ON}
  )
  add_custom_target(
      clang-tidy-hash
      COMMAND ${CMAKE_SOURCE_DIR}/tools/ClangTidyHash.sh
      ${CMAKE_BINARY_DIR}
      ${CMAKE_SOURCE_DIR}
      \${HASH}
  )
  set_target_properties(
      clang-tidy-hash
      PROPERTIES EXCLUDE_FROM_ALL TRUE
  )
  add_dependencies(
    clang-tidy-hash
    ${MODULES_TO_DEPEND_ON}
  )
else()
  message(STATUS "clang-tidy: Not using clang or couldn't find clang-tidy.")
endif()
