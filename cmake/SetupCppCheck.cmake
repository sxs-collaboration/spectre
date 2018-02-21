# Distributed under the MIT License.
# See LICENSE.txt for details.

if(NOT CPPCHECK_FOUND AND NOT CPPCHECK_EXECUTABLE)
  find_package(cppcheck QUIET)
endif()

if(CPPCHECK_EXECUTABLE)
  # The reason for checking if the output file .cppcheck_output.txt is
  # empty is because cppcheck check returns 0 even if some checks fail.
  # We do not use cppcheck's --error-exitcode flag because then cppcheck
  # returns the error code even if no issues are found (likely a
  # cppcheck bug).
  file(
    WRITE
    ${CMAKE_BINARY_DIR}/.cppcheck_run.sh
    "#!/bin/bash -e\n"
    "cppcheck "
    "  --inline-suppr "
    "  --template=\"[{severity}][{id}] {message} {callstack}\" "
    "  --enable=warning,performance,portability,information,missingInclude "
    "  --project=${CMAKE_BINARY_DIR}/compile_commands.json "
    "  --suppressions-list=${CMAKE_SOURCE_DIR}/tools/SuppressionsCppCheck.txt "
    "  -j2 2> ${CMAKE_BINARY_DIR}/.cppcheck_output.txt\n"
    "if [[ -s ${CMAKE_BINARY_DIR}/.cppcheck_output.txt ]]; then\n"
    "    echo 'cppcheck found errors:'\n"
    "    cat ${CMAKE_BINARY_DIR}/.cppcheck_output.txt\n"
    "    echo ''\n"
    "    exit 1\n"
    "fi\n"
    )
  execute_process(
    COMMAND chmod +x ${CMAKE_BINARY_DIR}/.cppcheck_run.sh
    RESULT_VARIABLE CHMOD_RESULT)
  if (NOT ${CHMOD_RESULT} EQUAL 0)
    message(FATAL_ERROR
      "Could not make ${CMAKE_BINARY_DIR}/.cppcheck_run.sh executable.")
  endif()
  # We run cppcheck on 2 cores since this is the maximum on TravisCI
  # and modern machines all have 2 cores.
  add_custom_target(
      cppcheck
      COMMAND
      ${CMAKE_BINARY_DIR}/.cppcheck_run.sh
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )
  set_target_properties(cppcheck PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
