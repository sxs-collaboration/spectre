# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(FUKA)

if (FUKA_FOUND)
  file(APPEND
    "${CMAKE_BINARY_DIR}/BuildInfo.txt"
    "FUKA: ${FUKA_ROOT}\n"
    )
endif()
