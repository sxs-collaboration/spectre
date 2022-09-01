## Distributed under the MIT License.
# See LICENSE.txt for details.

# These tests should get moved out of Unit eventually.

set(_COMPOSE_OUTPUT ${CMAKE_BINARY_DIR}/Test_ConvertComposeTableOutput.h5)

add_test(
  NAME "Unit.Executables.ConvertComposeTable"
  COMMAND
  ${SHELL_EXECUTABLE}
  -c
  "rm -f ${_COMPOSE_OUTPUT} && \
   ${CMAKE_BINARY_DIR}/bin/ConvertComposeTable \
   --compose-directory ${CMAKE_SOURCE_DIR}/tests/Unit/IO/ \
   --output ${_COMPOSE_OUTPUT} \
   --eos-subfile \"/EosTable\" \
   && rm ${_COMPOSE_OUTPUT}"
  )
set_standalone_test_properties("Unit.Executables.ConvertComposeTable")
set_tests_properties(
    "Unit.Executables.ConvertComposeTable"
    PROPERTIES
    LABELS "standalone;unit")
add_dependencies(
  unit-tests
  ConvertComposeTable
  )
add_dependencies(
  test-executables
  ConvertComposeTable
  )
