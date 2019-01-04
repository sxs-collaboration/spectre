# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(TCMALLOC REQUIRED)


spectre_include_directories(${TCMALLOC_INCLUDE_DIRS})
# Allocators should be linked as early as possible.
set(SPECTRE_LIBRARIES "${TCMALLOC_LIBRARIES};${SPECTRE_LIBRARIES}")

message(STATUS "tcmalloc libs: " ${TCMALLOC_LIBRARIES})
message(STATUS "tcmalloc incl: " ${TCMALLOC_INCLUDE_DIRS})

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "tcmalloc Version:  ${TCMALLOC_VERSION}\n"
  )
