# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(LIBXSMM REQUIRED)

spectre_include_directories(${LIBXSMM_INCLUDE_DIRS})
set(SPECTRE_LIBRARIES "${SPECTRE_LIBRARIES};${LIBXSMM_LIBRARIES}")

message(STATUS "LIBXSMM libs: " ${LIBXSMM_LIBRARIES})
message(STATUS "LIBXSMM incl: " ${LIBXSMM_INCLUDE_DIRS})
message(STATUS "LIBXSMM vers: " ${LIBXSMM_VERSION})

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "LIBXSMM Version:  ${LIBXSMM_VERSION}\n"
  )
