# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(JEMALLOC REQUIRED)

message(STATUS "jemalloc libs: " ${JEMALLOC_LIBRARIES})
message(STATUS "jemalloc incl: " ${JEMALLOC_INCLUDE_DIRS})
message(STATUS "jemalloc vers: " ${JEMALLOC_VERSION})

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "jemalloc Version:  ${JEMALLOC_VERSION}\n"
  )

add_library(Jemalloc INTERFACE IMPORTED)
set_property(TARGET Jemalloc
  APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${JEMALLOC_LIBRARIES})
set_property(TARGET Jemalloc PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${JEMALLOC_INCLUDE_DIRS})

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Jemalloc
  )
