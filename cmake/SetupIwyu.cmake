# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_iwyu_tool_targets IWYU_TOOL)
  # Run IWYU in parallel using half the available cores. On single
  # core machines using only 1 core
  include(ProcessorCount)
  ProcessorCount(N)
  math(EXPR N "${N} / 2")
  if(${N} EQUAL 0)
    set(N 1)
  endif()

  # IWYU for a single file
  add_custom_target(
    iwyu
    COMMAND ${IWYU_TOOL}
    -j ${N}
    -p ${CMAKE_BINARY_DIR}
    \${FILE}
    --
    --check_also=\${CHECK_ALSO}
    --mapping_file=${CMAKE_SOURCE_DIR}/tools/Iwyu/iwyu.imp
    )
  add_dependencies(
    iwyu
    ${MODULES_TO_DEPEND_ON}
    )
  set_target_properties(
    iwyu
    PROPERTIES EXCLUDE_FROM_ALL TRUE
    )

  # IWYU for all files modified between two hashes
  add_custom_target(
    iwyu-hash
    COMMAND
    ${CMAKE_SOURCE_DIR}/.travis/RunIncludeWhatYouUse.sh
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}
    ${IWYU_TOOL}
    ${N}
    \${HASH}
    )
  add_dependencies(
    iwyu-hash
    ${MODULES_TO_DEPEND_ON}
    )
  set_target_properties(
    iwyu-hash
    PROPERTIES EXCLUDE_FROM_ALL TRUE
    )
endfunction(add_iwyu_tool_targets IWYU_TOOL)

if(NOT ${USE_PCH})
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(IWYU_REQUIRED_VERSION 0.9)
    find_program(IWYU_BINARY include-what-you-use)
    find_program(IWYU_TOOL iwyu_tool.py)
    if("${IWYU_TOOL}" STREQUAL "IWYU_TOOL-NOTFOUND")
      message(STATUS "Could not find include-what-you-use iwyu_tool.py")
    else("${IWYU_TOOL}" STREQUAL "IWYU_TOOL-NOTFOUND")
      # Parse IWYU version
      execute_process(
        COMMAND ${IWYU_BINARY} --version
        OUTPUT_VARIABLE IWYU_VERSION_STRING
        )
      string(
        REPLACE "include-what-you-use " ""
        IWYU_VERSION_STRING "${IWYU_VERSION_STRING}")
      string(FIND ${IWYU_VERSION_STRING} " " IWYU_VERSION_END)
      string(SUBSTRING ${IWYU_VERSION_STRING} 0 ${IWYU_VERSION_END} IWYU_VERSION)
      if(${IWYU_VERSION} VERSION_LESS ${IWYU_REQUIRED_VERSION})
        message(STATUS
          "include-what-you-use version must be ${IWYU_REQUIRED_VERSION} "
          "but found ${IWYU_VERSION}")
        set(IWYU_TOOL "IWYU_TOOL-NOTFOUND")
      else(${IWYU_VERSION} VERSION_LESS ${IWYU_REQUIRED_VERSION})
        message(STATUS "iwyu include-what-you-use: ${IWYU_BINARY}")
        message(STATUS "iwyu iwyu_tool.py: ${IWYU_TOOL}")
        message(STATUS "iwyu vers: ${IWYU_VERSION}")
        add_iwyu_tool_targets(${IWYU_TOOL})
      endif(${IWYU_VERSION} VERSION_LESS ${IWYU_REQUIRED_VERSION})
    endif("${IWYU_TOOL}" STREQUAL "IWYU_TOOL-NOTFOUND")
  else(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS
      "iwyu: Cannot use include-what-you-use without clang compiler. "
      "Compile with clang to use iwyu.")
  endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
else(NOT ${USE_PCH})
  message(STATUS
    "iwyu: Cannot use include-what-you-use with precompiled header. "
    "Pass USE_PCH=OFF to CMake to disable the precompiled header.")
endif(NOT ${USE_PCH})
