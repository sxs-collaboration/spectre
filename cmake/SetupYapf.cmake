# Distributed under the MIT License.
# See LICENSE.txt for details.

find_program(YAPF_EXECUTABLE yapf PATHS ENV PATH)

if (YAPF_EXECUTABLE)
  execute_process(
    COMMAND ${YAPF_EXECUTABLE} --version
    OUTPUT_VARIABLE YAPF_VERSION)
  message(STATUS "yapf found at: ${YAPF_EXECUTABLE}")
  message(STATUS "yapf version: ${YAPF_VERSION}")
else()
  message(STATUS
    "yapf not available. Please install "
    "YAPF (https://github.com/google/yapf)")
endif()
