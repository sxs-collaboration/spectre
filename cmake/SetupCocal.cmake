# Distributed under the MIT License.
# See LICENSE.txt for details.


find_package(COCAL)

if (COCAL_FOUND)
  message("-- Found Cocal at ${COCAL_ROOT}")
  file(APPEND
    "${CMAKE_BINARY_DIR}/BuildInfo.txt"
    "COCAL: ${COCAL_ROOT}\n"
    )
else()
  message("-- Could not find Cocal")
endif()
