# Distributed under the MIT License.
# See LICENSE.txt for details.

# Always create the target and global property so that we don't need to
# special case elsewhere in the code.
set_property(GLOBAL
  PROPERTY
  SPECTRE_ALLOCATOR_LIBRARY
  "")

add_library(SpectreAllocator INTERFACE)

# Sanitizers are not guaranteed to work with custom malloc
# since they already intercept malloc calls.
if(NOT ${ASAN})
  option(MEMORY_ALLOCATOR
    "Which allocator to use: SYSTEM, TCMALLOC, JEMALLOC (default)"
    OFF)
  # We need to link custom allocators before we link anything else so that
  # any third-party libraries, which generally should all be built as shared
  # libraries, use the allocator that we use. Unfortunately, how exactly
  # CMake decides on the linking order is not clear when using
  # INTERFACE_LINK_LIBRARIES and targets. To this end, we set a global
  # property SPECTRE_ALLOCATOR_LIBRARY that contains the link flag to link
  # to the memory allocator. By linking to the allocator library first
  # explicitly in target_link_libraries CMake correctly places the allocator
  # library as the first entry in the link libraries. We also link to the
  # SpectreAllocator target to pull in any additional allocator-related
  # flags, such as include directories.
  #
  # Targets can grab the link flags using:
  #
  #   get_property(
  #     SPECTRE_ALLOCATOR_LIBRARY
  #     GLOBAL
  #     PROPERTY SPECTRE_ALLOCATOR_LIBRARY
  #     )
  #   target_link_libraries(${TARGET_NAME}
  #     PUBLIC
  #     ${SPECTRE_ALLOCATOR_LIBRARY}
  #     SpectreAllocator
  #     )
  #
  # These need to be the first call to target_link_libraries of the target.
  if("${MEMORY_ALLOCATOR}" STREQUAL "JEMALLOC"
      OR "${MEMORY_ALLOCATOR}" STREQUAL "OFF")
    include(SetupJemalloc)
    target_link_libraries(
      SpectreAllocator
      INTERFACE
      Jemalloc
      )
    get_property(
      SPECTRE_ALLOCATOR_LIBRARY
      TARGET Jemalloc
      PROPERTY INTERFACE_LINK_LIBRARIES
      )
    set_property(GLOBAL
      PROPERTY
      SPECTRE_ALLOCATOR_LIBRARY
      ${SPECTRE_ALLOCATOR_LIBRARY})
  elseif("${MEMORY_ALLOCATOR}" STREQUAL "TCMALLOC")
    include(SetupTcmalloc)
    target_link_libraries(
      SpectreAllocator
      INTERFACE
      Tcmalloc
      )
    get_property(
      SPECTRE_ALLOCATOR_LIBRARY
      TARGET Tcmalloc
      PROPERTY INTERFACE_LINK_LIBRARIES
      )
    set_property(GLOBAL
      PROPERTY
      SPECTRE_ALLOCATOR_LIBRARY
      ${SPECTRE_ALLOCATOR_LIBRARY})
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
