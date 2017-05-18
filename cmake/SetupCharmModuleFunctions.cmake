# Distributed under the MIT License.
# See LICENSE.txt for details.

# Add a function to generate the charm interface files for the mainmodule.
function(add_charm_mainmodule MAIN_MODULE)
  # Arguments:
  #   MAIN_MODULE: Name of the Charm++ mainmodule.
  add_custom_command(
      OUTPUT ${MAIN_MODULE}.decl.h ${MAIN_MODULE}.def.h
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${MAIN_MODULE}.ci
      COMMAND ${CHARM_COMPILER} -no-charmrun
      ${CMAKE_SOURCE_DIR}/${MAIN_MODULE}.ci
  )
endfunction()

# Add a function to generate the charm interface files for a module.
function(add_charm_module MODULE)
  # Arguments:
  #   MODULE: Name of the Charm++ module.
  add_charm_mainmodule(${MODULE})
  add_custom_target(
      module_${MODULE}
      DEPENDS ${MODULE}.decl.h ${MODULE}.def.h
  )
endfunction()

# Add a function to generate the charm interface files for the mainmodule for
# evolution systems. The reason we need a separate a separate function for
# generated systems is that for these the interface (ci) files are located in
# the build directory as opposed to the source directory.
function(add_generated_charm_mainmodule MAIN_MODULE)
  # Arguments:
  #   MAIN_MODULE: Name of the Charm++ mainmodule.
  add_custom_command(
      OUTPUT ${MAIN_MODULE}.decl.h ${MAIN_MODULE}.def.h
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${MAIN_MODULE}.ci
      COMMAND ${CHARM_COMPILER} -no-charmrun
      ${CMAKE_BINARY_DIR}/${MAIN_MODULE}.ci
  )
endfunction()
