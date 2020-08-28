# Distributed under the MIT License.
# See LICENSE.txt for details.

# Remove the checks because:
#
# Notes:
# misc-move-const-arg: we keep this check because even though this gives
#                      a lot of annoying warnings about moving trivially
#                      copyable types, it warns about moving const objects,
#                      which can have severe performance impacts.
set(CLANG_TIDY_IGNORE_CHECKS "*,"
    "-android-*,"
    "-bugprone-exception-escape,"
    # complains about Catch macros having infinite loops
    "-bugprone-infinite-loop,"
    "-bugprone-macro-parentheses,"
    # many static variables we use do not throw, and if they do we
    # want to terminate anyway
    "-cert-err58-cpp,"
    "-cert-msc51-cpp,"
    "-cert-msc32-c,"
    # checks for incorrectly implemented self-assignment
    # checks.  However, it's broken.
    "-cert-oop54-cpp,"
    "-cppcoreguidelines-avoid-c-arrays,"
    # too many inconvenient positives for us to deal with
    "-cppcoreguidelines-avoid-magic-numbers,"
    # false positives
    "-cppcoreguidelines-c-copy-assignment-signature,"
    # sometimes macros are the right answer.
    "-cppcoreguidelines-macro-usage,"
    # we want to use malloc with unique_ptr because not initializing
    # the memory is faster.
    "-cppcoreguidelines-no-malloc,"
    # public and protected member variables are fine
    "-cppcoreguidelines-non-private-member-variables-in-classes,"
    "-fuchsia-*,"
    # defaulting virtual functions in CoordinateMap
    "-google-default-arguments,"
    # specifying int32_t and int64_t instead of just int
    "-google-runtime-int,"
    # redundant with other checks
    "-hicpp-*,"
    # We use pragma once instead of include guards
    "-llvm-header-guard,"
    # Makes code less portable because some implementation-defined STL
    # types can be pointers or not.  (Same as
    # readability-qualified-auto below)
    "-llvm-qualified-auto,"
    # thinks constexpr variables in header files cause ODR violations
    "-misc-definitions-in-headers,"
    # false positives
    "-misc-noexcept-move-constructor,"
    "-misc-non-private-member-variables-in-classes,"
    # false positives
    "-misc-unconventional-assign-operator,"
    "-modernize-raw-string-literal,"
    # should be used, but requires possibly a lot of code changes that
    # we don't have the resources for
    "-modernize-use-nodiscard,"
    # this wants everything to use trailing return type syntax, which
    # is silly.
    "-modernize-use-trailing-return-type,"
    "-performance-noexcept-move-constructor,"
    # complains about decltype(auto)
    "-readability-const-return-type,"
    # style choice, discussed in issue #145
    "-readability-else-after-return,"
    "-readability-magic-numbers,"
    # Same as llvm-qualified-auto above.
    "-readability-qualified-auto,"
    # We can have two of: methods are static when possible, static
    # methods are not called through instances, and methods of
    # calling, e.g., x.size(), are consistent across classes.  We
    # choose to lose this one.
    "-readability-static-accessed-through-instance,"
    # we're okay with lower case
    "-readability-uppercase-literal-suffix,"
    )
string(REPLACE ";" "" CLANG_TIDY_IGNORE_CHECKS "${CLANG_TIDY_IGNORE_CHECKS}")

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
    module_All
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
