# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(JEMALLOC REQUIRED)

message(STATUS "jemalloc libs: " ${JEMALLOC_LIBRARIES})
message(STATUS "jemalloc incl: " ${JEMALLOC_INCLUDE_DIRS})
message(STATUS "jemalloc vers: " ${JEMALLOC_VERSION})

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
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

# Determine whether we are using JEMALLOC as a shared or static library.
# Since libraries can be named, e.g. libjemalloc.so.2 we just search for
# the extension after stripping the directories. It's unlikely jemalloc
# will rename themselves to "alloc.so" and then we'd have libs like
# "alloc.so.a"
get_filename_component(
  JEMALLOC_LIB_NAME
  ${JEMALLOC_LIBRARIES}
  NAME
  )
if(APPLE)
  string(FIND "${JEMALLOC_LIB_NAME}" ".dylib" FOUND_SHARED)
else()
  string(FIND "${JEMALLOC_LIB_NAME}" ".so" FOUND_SHARED)
endif()
if(${FOUND_SHARED} STREQUAL -1)
  string(FIND "${JEMALLOC_LIB_NAME}" ".a" FOUND_STATIC)
  if(${FOUND_STATIC} EQUAL -1)
    message(FATAL_ERROR "Failed to determine whether JEMALLOC is a shared or "
      "static library. Found: ${JEMALLOC_LIBRARIES}")
  endif()
  set(JEMALLOC_LIB_TYPE STATIC)
else()
  set(JEMALLOC_LIB_TYPE SHARED)
endif()

mark_as_advanced(JEMALLOC_LIB_TYPE)
