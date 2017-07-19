# Distributed under the MIT License.
# See LICENSE.txt for details.

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

# Remove the checks because:
# -llvm-header-guard: We use pragma once instead of include guards
# -google-runtime-int: specifying int32_t and int64_t instead of just int
if (CLANG_TIDY_BIN)
  add_custom_target(
      clang-tidy
      COMMAND
      ${CLANG_TIDY_BIN}
      -header-filter=${CMAKE_SOURCE_DIR}
      -checks=*,-llvm-header-guard,-google-runtime-int
      -p ${CMAKE_BINARY_DIR}
      \${FILE}
  )
  set_target_properties(
      clang-tidy
      PROPERTIES EXCLUDE_FROM_ALL TRUE
  )
endif()
