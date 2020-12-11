# Distributed under the MIT License.
# See LICENSE.txt for details.

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
      --quiet
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
      \${NUM_THREADS}
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
      \${NUM_THREADS}
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
