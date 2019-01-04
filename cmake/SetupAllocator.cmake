# Distributed under the MIT License.
# See LICENSE.txt for details.

# Sanitizers are not guaranteed to work with custom malloc
# since they already intercept malloc calls.
if(NOT ${ASAN})
  option(MEMORY_ALLOCATOR
    "Which allocator to use: SYSTEM, TCMALLOC, JEMALLOC (default)"
    OFF)

  if("${MEMORY_ALLOCATOR}" STREQUAL "JEMALLOC"
      OR "${MEMORY_ALLOCATOR}" STREQUAL "OFF")
    include(SetupJemalloc)
  elseif("${MEMORY_ALLOCATOR}" STREQUAL "TCMALLOC")
    include(SetupTcmalloc)
  elseif(NOT "${MEMORY_ALLOCATOR}" STREQUAL "SYSTEM")
    message(FATAL_ERROR
      "Unknown memory allocator specified '${MEMORY_ALLOCATOR}'. "
      "Known options are:\n"
      "  SYSTEM, TCMALLOC, JEMALLOC (default)")
  else()
    message(STATUS "Using system default memory allocator.")
  endif()
else(NOT ${ASAN})
  message(STATUS
    "Using system default malloc since we are using address sanitizer which "
    "may have issues when using a custom allocator.")
endif(NOT ${ASAN})
