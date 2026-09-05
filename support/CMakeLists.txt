# Distributed under the MIT License.
# See LICENSE.txt for details.

# The selected machine's support files go into 'support/' next to 'bin/' and
# 'lib/'. A simulation gets a frozen copy of them; see 'BinDirectory.py'. If you
# change this list, also reconsider which files are copied to bin directories
# (edit 'BinDirectory.SUPPORT_FILES').
if(MACHINE)
  message(STATUS "Selected machine: ${MACHINE}")
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/Machines/${MACHINE}.yaml
    ${CMAKE_BINARY_DIR}/support/Machine.yaml
    )
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/SubmitScripts/SubmitTemplateBase.sh
    ${CMAKE_BINARY_DIR}/support/SubmitTemplateBase.sh
    @ONLY
    )
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/SubmitScripts/${MACHINE}.sh
    ${CMAKE_BINARY_DIR}/support/SubmitTemplate.sh
    @ONLY
    )
endif()

add_subdirectory(Pipelines)
add_subdirectory(Python)
