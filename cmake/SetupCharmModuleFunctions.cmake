# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(PythonInterp REQUIRED)

# Add a function to generate the charm interface files for the module.
function(add_charm_module MODULE)
  # Arguments:
  #   MODULE: Name of the Charm++ module.

  if (NOT TARGET module_All)
    # Target that will have all the Charm modules as dependencies
    add_custom_target(module_All)
  endif()

  add_custom_command(
    OUTPUT ${MODULE}.decl.h ${MODULE}.def.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}.ci
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}.ci ${MODULE}.ci
    COMMAND ${CHARM_COMPILER} -no-charmrun ${MODULE}.ci
    )
  add_custom_target(
    module_${MODULE}
    DEPENDS ${MODULE}.decl.h ${MODULE}.def.h
    )
  add_dependencies(
    module_All
    module_${MODULE}
    )
endfunction()
