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
    if(NOT ${TARGET_IS_IMPORTED} AND TARGET SpectrePch)
      target_precompile_headers(${TARGET_NAME} REUSE_FROM SpectrePch)
      target_link_libraries(${TARGET_NAME} PRIVATE SpectrePchFlags)
    endif()
  endif (USE_PCH)
endfunction()
