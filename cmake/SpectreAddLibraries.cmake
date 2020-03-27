# Distributed under the MIT License.
# See LICENSE.txt for details.

add_custom_target(libs)

function(ADD_SPECTRE_LIBRARY LIBRARY_NAME)
  add_library(${LIBRARY_NAME} ${ARGN})
  add_dependencies(libs ${LIBRARY_NAME})
  set_target_properties(
    ${TARGET_NAME}
    PROPERTIES
    RULE_LAUNCH_LINK "${CMAKE_BINARY_DIR}/tmp/WrapLibraryLinker.sh"
    LINK_DEPENDS "${CMAKE_BINARY_DIR}/tmp/WrapLibraryLinker.sh"
    )

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
  if (NOT "${LIBRARY_NAME}" MATCHES "^${SPECTRE_PCH}"
      AND NOT ${LIBRARY_IS_IMPORTED}
      AND NOT ${LIBRARY_TYPE} STREQUAL INTERFACE_LIBRARY)
    add_dependencies(${LIBRARY_NAME} ${SPECTRE_PCH})
    set_source_files_properties(
        ${ARGN}
        OBJECT_DEPENDS "${SPECTRE_PCH_PATH}"
        )
    target_compile_options(
      ${LIBRARY_NAME}
      PRIVATE
      $<TARGET_PROPERTY:${SPECTRE_PCH},INTERFACE_COMPILE_OPTIONS>
      )
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
  endif()
endfunction()
