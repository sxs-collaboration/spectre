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
endfunction()
