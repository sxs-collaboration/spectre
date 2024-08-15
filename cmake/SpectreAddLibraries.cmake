# Distributed under the MIT License.
# See LICENSE.txt for details.

add_custom_target(libs)

function(ADD_SPECTRE_LIBRARY LIBRARY_NAME)
  add_library(${LIBRARY_NAME} ${ARGN})
  add_dependencies(libs ${LIBRARY_NAME})

  get_target_property(
    LIBRARY_IS_IMPORTED
    ${LIBRARY_NAME}
    IMPORTED
    )
  get_target_property(
    LIBRARY_TYPE
    ${LIBRARY_NAME}
    TYPE
    )

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
  get_property(
    SPECTRE_ALLOCATOR_LIBRARY
    GLOBAL
    PROPERTY SPECTRE_ALLOCATOR_LIBRARY
    )
  if (NOT ${LIBRARY_TYPE} STREQUAL INTERFACE_LIBRARY)
    target_link_libraries(${LIBRARY_NAME}
      PUBLIC
      ${SPECTRE_ALLOCATOR_LIBRARY}
      SpectreAllocator
    )

    set(SPECTRE_KOKKOS_LAUNCHER "")
    if(SPECTRE_KOKKOS)
      # We need to make sure we don't drop the Kokkos link wrapper
      get_target_property(
        _RULE_LAUNCH_LINK
        ${LIBRARY_NAME}
        RULE_LAUNCH_LINK)
      if (_RULE_LAUNCH_LINK)
        set(SPECTRE_KOKKOS_LAUNCHER ${_RULE_LAUNCH_LINK})
      endif()
    endif()
    set_target_properties(
      ${LIBRARY_NAME}
      PROPERTIES
      RULE_LAUNCH_LINK
      "${CMAKE_BINARY_DIR}/tmp/WrapLibraryLinker.sh ${SPECTRE_KOKKOS_LAUNCHER}"
      LINK_DEPENDS "${CMAKE_BINARY_DIR}/tmp/WrapLibraryLinker.sh"
    )
  endif (NOT ${LIBRARY_TYPE} STREQUAL INTERFACE_LIBRARY)
  if (NOT "${LIBRARY_NAME}" MATCHES "^SpectrePch"
      AND NOT ${LIBRARY_IS_IMPORTED}
      AND NOT ${LIBRARY_TYPE} STREQUAL INTERFACE_LIBRARY
      AND TARGET SpectrePch)
      target_precompile_headers(${LIBRARY_NAME} REUSE_FROM SpectrePch)
      target_link_libraries(${LIBRARY_NAME} PRIVATE SpectrePchFlags)
  endif()
  if (${LIBRARY_TYPE} STREQUAL INTERFACE_LIBRARY)
    target_link_libraries(
      ${LIBRARY_NAME}
      INTERFACE
      SpectreFlags
      )
  else()
    target_link_libraries(
      ${LIBRARY_NAME}
      PUBLIC
      SpectreFlags
      )
    set_property(
      TARGET ${LIBRARY_NAME}
      PROPERTY FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
      )
  endif()
  if (BUILD_SHARED_LIBS
      AND NOT ${LIBRARY_TYPE} STREQUAL INTERFACE_LIBRARY)
    install(TARGETS ${LIBRARY_NAME} OPTIONAL
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
endfunction()
