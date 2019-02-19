# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_spectre_executable TARGET_NAME)
  _add_spectre_executable(${TARGET_NAME} ${ARGN})

  if (USE_PCH)
    add_dependencies(${TARGET_NAME} pch)
    set_source_files_properties(
      ${ARGN}
      OBJECT_DEPENDS "${PCH_PATH};${PCH_PATH}.gch"
      )
    get_target_property(
      TARGET_IS_IMPORTED
      ${TARGET_NAME}
      IMPORTED
      )
    if(NOT ${TARGET_IS_IMPORTED})
      target_compile_options(
        ${TARGET_NAME}
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${PCH_FLAG}>
        )
    endif()
  endif (USE_PCH)
endfunction()
