# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(COCAL REQUIRED)

if (COCAL_FOUND)
  message("Found cocal")
  file(APPEND
    "${CMAKE_BINARY_DIR}/BuildInfo.txt"
    "COCAL: ${COCAL_ROOT}\n"
    )
endif()
