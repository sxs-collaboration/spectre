# Distributed under the MIT License.
# See LICENSE.txt for details.

# Place support files in bin/ so they are available in copied or linked
# simulation bin directories
option(MACHINE "Select a machine that we know how to run on, such as a \
particular supercomputer" OFF)
if(MACHINE)
  message(STATUS "Selected machine: ${MACHINE}")
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/Machines/${MACHINE}.yaml
    ${CMAKE_BINARY_DIR}/bin/Machine.yaml
    )
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/SubmitScripts/SubmitTemplateBase.sh
    ${CMAKE_BINARY_DIR}/bin/SubmitTemplateBase.sh
    @ONLY
    )
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/SubmitScripts/${MACHINE}.sh
    ${CMAKE_BINARY_DIR}/bin/SubmitTemplate.sh
    @ONLY
    )
endif()

add_subdirectory(Pipelines)
add_subdirectory(Python)
