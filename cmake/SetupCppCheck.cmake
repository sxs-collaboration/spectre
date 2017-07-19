# Distributed under the MIT License.
# See LICENSE.txt for details.

if(NOT CPPCHECK_FOUND AND NOT CPPCHECK_EXECUTABLE)
  find_package(cppcheck QUIET)
endif()

if(CPPCHECK_EXECUTABLE)
  # We run cppcheck on 2 cores since this is the maximum on TravisCI
  # and modern machines all have 2 cores.
  add_custom_target(
      cppcheck
      COMMAND
      ${CPPCHECK_EXECUTABLE}
      --inline-suppr
      --template=\"[{severity}][{id}] {message} {callstack}\"
      --enable=warning,performance,portability,information,missingInclude
      --project=${CMAKE_BINARY_DIR}/compile_commands.json
      --suppressions-list=${CMAKE_SOURCE_DIR}/tools/SuppressionsCppCheck.txt
      -j2

      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )
  set_target_properties(cppcheck PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
