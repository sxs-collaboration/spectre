# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Blaze 3.2 REQUIRED)

spectre_include_directories("${BLAZE_INCLUDE_DIR}")
message(STATUS "Blaze incl: ${BLAZE_INCLUDE_DIR}")
message(STATUS "Blaze vers: ${BLAZE_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Blaze Version:  ${BLAZE_VERSION}\n"
  )
