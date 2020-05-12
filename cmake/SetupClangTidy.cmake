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
# -modernize-use-trailing-return-type: this wants everything to use trailing
#                                      return type syntax, which is silly.
# -cert-oop54-cpp: checks for incorrectly implemented self-assignment checks.
#                  However, it's broken.
# -misc-definitions-in-headers: thinks constexpr variables in header files
#                               cause ODR violations
# -modernize-use-nodiscard: while it would be great to do this, changing
#                           the entire source tree is something we don't have
#                           the resources for.
# -hicpp-*: redundant with other checks
# -cppcoreguidelines-macro-usage: sometimes macros are the right answer.
# -cppcoreguidelines-avoid-magic-numbers: too many inconvenient positives
#                                         for us to deal with
# -modernize-use-nodiscard: should be used, but requires possibly a lot of code
#                           changes that we don't have the resources for
# -cppcoreguidelines-non-private-member-variables-in-classes:
#         public and protected member variables are fine
# -readability-uppercase-literal-suffix: we're okay with lower case
#
# Notes:
# misc-move-const-arg: we keep this check because even though this gives
#                      a lot of annoying warnings about moving trivially
#                      copyable types, it warns about moving const objects,
#                      which can have severe performance impacts.
set(CLANG_TIDY_IGNORE_CHECKS "*,-cppcoreguidelines-no-malloc,-llvm-header-guard,-google-runtime-int,-readability-else-after-return,-misc-noexcept-move-constructor,-misc-unconventional-assign-operator,-cppcoreguidelines-c-copy-assignment-signature,-modernize-raw-string-literal,-hicpp-*,-android-*,-cert-err58-cpp,-google-default-arguments,-fuchsia-*,-performance-noexcept-move-constructor,-modernize-use-trailing-return-type,-cert-oop54-cpp,-misc-definitions-in-headers,-cppcoreguidelines-macro-usage,-cppcoreguidelines-avoid-magic-numbers,-modernize-use-nodiscard,-readability-magic-numbers,-bugprone-exception-escape,-cert-msc32-c,-misc-non-private-member-variables-in-classes,-cppcoreguidelines-avoid-c-arrays,-cppcoreguidelines-non-private-member-variables-in-classes,-cert-msc51-cpp,-bugprone-macro-parentheses,-readability-uppercase-literal-suffix")

if(NOT CLANG_TIDY_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(CLANG_TIDY_ROOT "")
  set(CLANG_TIDY_ROOT $ENV{CLANG_TIDY_ROOT})
endif()

if(NOT CMAKE_CXX_CLANG_TIDY AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(
      REGEX MATCH "^[0-9]+.[0-9]+" LLVM_VERSION
      "${CMAKE_CXX_COMPILER_VERSION}"
  )
  find_program(
      CLANG_TIDY_BIN
      NAMES "clang-tidy-${LLVM_VERSION}" "clang-tidy"
      HINTS ${CLANG_TIDY_ROOT}
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
  if (TARGET SpectrePch)
    list(APPEND MODULES_TO_DEPEND_ON SpectrePch)
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
      # Make the build target fail when any warnings occur
      -warnings-as-errors=*
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
