# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(SpEC)

if (SpEC_FOUND)
  file(APPEND
    "${CMAKE_BINARY_DIR}/BuildInfo.txt"
    "SpEC exporter: ${SPEC_EXPORTER_ROOT}\n"
    )
endif()
