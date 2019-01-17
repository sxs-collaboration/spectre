# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(JEMALLOC REQUIRED)


spectre_include_directories(${JEMALLOC_INCLUDE_DIRS})
# Allocators should be linked as early as possible.
set(SPECTRE_LIBRARIES "${JEMALLOC_LIBRARIES};${SPECTRE_LIBRARIES}")

message(STATUS "jemalloc libs: " ${JEMALLOC_LIBRARIES})
message(STATUS "jemalloc incl: " ${JEMALLOC_INCLUDE_DIRS})
message(STATUS "jemalloc vers: " ${JEMALLOC_VERSION})

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "jemalloc Version:  ${JEMALLOC_VERSION}\n"
  )
