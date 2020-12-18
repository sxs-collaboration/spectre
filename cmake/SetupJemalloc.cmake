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

# Determine whether we are using JEMALLOC as a shared or static library
get_filename_component(
  JEMALLOC_LIB_NAME
  ${JEMALLOC_LIBRARIES}
  NAME
  )
get_filename_component(
  JEMALLOC_LIB_TYPE
  ${JEMALLOC_LIB_NAME}
  LAST_EXT
  )
while(NOT "${JEMALLOC_LIB_TYPE}" STREQUAL ".so"
    AND NOT "${JEMALLOC_LIB_TYPE}" STREQUAL ".a"
    AND NOT "${JEMALLOC_LIB_TYPE}" STREQUAL ".dylib")
  get_filename_component(
    JEMALLOC_LIB_NAME
    ${JEMALLOC_LIB_NAME}
    NAME_WLE
    )
  get_filename_component(
    JEMALLOC_LIB_TYPE
    ${JEMALLOC_LIB_NAME}
    LAST_EXT
    )
endwhile()

if("${JEMALLOC_LIB_TYPE}" STREQUAL ".a")
  set(JEMALLOC_LIB_TYPE STATIC)
elseif("${JEMALLOC_LIB_TYPE}" STREQUAL ".so" OR
    "${JEMALLOC_LIB_TYPE}" STREQUAL ".dylib")
  set(JEMALLOC_LIB_TYPE SHARED)
else()
  message(FATAL_ERROR "Couldn't determine whether JEMALLOC is a static "
    "or shared/dynamic library.")
endif()
mark_as_advanced(JEMALLOC_LIB_TYPE)
