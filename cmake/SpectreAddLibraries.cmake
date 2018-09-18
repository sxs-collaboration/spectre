# Distributed under the MIT License.
# See LICENSE.txt for details.

add_custom_target(libs)

function(ADD_SPECTRE_LIBRARY LIBRARY_NAME)
  add_library(${LIBRARY_NAME} ${ARGN})
  add_dependencies(libs ${LIBRARY_NAME})
endfunction()
