# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_spectre_executable TARGET_NAME)
  _add_spectre_executable(${TARGET_NAME} ${ARGN})

  if (USE_PCH)
    get_target_property(
      TARGET_IS_IMPORTED
      ${TARGET_NAME}
      IMPORTED
      )
    if(NOT ${TARGET_IS_IMPORTED})
      add_dependencies(${TARGET_NAME} ${SPECTRE_PCH})
      set_source_files_properties(
        ${ARGN}
        OBJECT_DEPENDS "${SPECTRE_PCH_PATH}"
        )
      target_compile_options(
        ${TARGET_NAME}
        PRIVATE
        $<TARGET_PROPERTY:${SPECTRE_PCH},INTERFACE_COMPILE_OPTIONS>
        )
    endif()
  endif (USE_PCH)
endfunction()
