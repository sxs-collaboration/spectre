# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Catch 2.1.0 REQUIRED)

spectre_include_directories("${CATCH_INCLUDE_DIR}")
message(STATUS "Catch include: ${CATCH_INCLUDE_DIR}")
message(STATUS "Catch version: ${CATCH_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Catch Version:  ${CATCH_VERSION}\n"
  )
