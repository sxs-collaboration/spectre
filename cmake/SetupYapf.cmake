# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Yapf)

if (YAPF_EXECUTABLE)
  message(STATUS "yapf found at: ${YAPF_EXECUTABLE}")
  message(STATUS "yapf version: ${YAPF_VERSION}")
else()
  message(STATUS
    "yapf not available. Please install "
    "YAPF (https://github.com/google/yapf)")
endif()
