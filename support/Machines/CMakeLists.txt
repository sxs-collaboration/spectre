# Distributed under the MIT License.
# See LICENSE.txt for details.

option(MACHINE "Select a machine that we know how to run on, such as a \
particular supercomputer" OFF)

spectre_python_add_module(
  support
  PYTHON_FILES
  Machines.py
)

if(MACHINE)
  message(STATUS "Selected machine: ${MACHINE}")
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/Machines/${MACHINE}.yaml
    ${SPECTRE_PYTHON_PREFIX}/support/Machine.yaml
    )
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/SubmitScripts/${MACHINE}.sh
    ${SPECTRE_PYTHON_PREFIX}/support/Submit.sh
    )
endif()
