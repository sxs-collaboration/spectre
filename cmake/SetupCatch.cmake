# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Catch2 3.0.0 REQUIRED)

message(STATUS "Catch version: ${Catch2_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "Catch version: ${Catch2_VERSION}\n"
  )
