# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Catch 2.8.0 REQUIRED)

spectre_include_directories("${CATCH_INCLUDE_DIR}")
message(STATUS "Catch include: ${CATCH_INCLUDE_DIR}")
message(STATUS "Catch version: ${CATCH_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "Catch version: ${CATCH_VERSION}\n"
  )
