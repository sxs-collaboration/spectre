# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(JEMALLOC REQUIRED)


spectre_include_directories(${JEMALLOC_INCLUDE_DIRS})
set(SPECTRE_LIBRARIES "${SPECTRE_LIBRARIES};${JEMALLOC_LIBRARIES}")

message(STATUS "jemalloc libs: " ${JEMALLOC_LIBRARIES})
message(STATUS "jemalloc incl: " ${JEMALLOC_INCLUDE_DIRS})
message(STATUS "jemalloc vers: " ${JEMALLOC_VERSION})

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "jemalloc Version:  ${JEMALLOC_VERSION}\n"
  )
