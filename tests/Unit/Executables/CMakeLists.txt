## Distributed under the MIT License.
# See LICENSE.txt for details.

# These tests should get moved out of Unit eventually.

set(_COMPOSE_OUTPUT ${CMAKE_BINARY_DIR}/Test_ConvertComposeTableOutput.h5)
set(_CONNECTIVITY_OUTPUT ${CMAKE_BINARY_DIR}/Test_ExtendConnectivityOutput)

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

add_test(
  NAME "Unit.Executables.ExtendConnectivity"
  COMMAND
  ${SHELL_EXECUTABLE}
  -c
  "rm -f ${_CONNECTIVITY_OUTPUT} && \
   cp ${CMAKE_SOURCE_DIR}/tests/Unit/IO/Test_ExtendConnectivity_Data0.h5 \
   ${CMAKE_BINARY_DIR}/tmp && \
   ${CMAKE_BINARY_DIR}/bin/ExtendConnectivity \
   --file_name ${CMAKE_BINARY_DIR}/tmp/Test_ExtendConnectivity_Data0.h5 \
   --subfile_name VolumeData && \
   ${CMAKE_BINARY_DIR}/bin/spectre generate-xdmf \
   --output ${_CONNECTIVITY_OUTPUT} \
   --subfile-name VolumeData \
   ${CMAKE_BINARY_DIR}/tmp/Test_ExtendConnectivity_Data0.h5 \
   && rm ${CMAKE_BINARY_DIR}/tmp/Test_ExtendConnectivity_Data0.h5 \
   && rm ${_CONNECTIVITY_OUTPUT}.xmf"
  )
set_standalone_test_properties("Unit.Executables.ExtendConnectivity")
add_dependencies(
  test-executables
  ExtendConnectivity
  )
