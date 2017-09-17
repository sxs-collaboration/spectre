# Distributed under the MIT License.
# See LICENSE.txt for details.

# Add a function to generate the charm interface files for the module.
function(add_charm_module MODULE)
  # Arguments:
  #   MODULE: Name of the Charm++ module.
  add_custom_command(
      OUTPUT ${MODULE}.decl.h ${MODULE}.def.h
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}.ci
      COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}.ci ${MODULE}.ci

      COMMAND perl -i -p00e 's/typename[\\s]+\([A-Za-z]\)\(?=[^\(]*\\\)\)/__typename__\\1/g' ${MODULE}.ci

      COMMAND perl -pi -e 's/std::array/__std____array__/g' ${MODULE}.ci

      COMMAND ${CHARM_COMPILER} -no-charmrun ${MODULE}.ci

      COMMAND perl -pi -e 's/__typename__/typename /g' ${MODULE}.decl.h
      COMMAND perl -pi -e 's/__typename__/typename /g' ${MODULE}.def.h
      COMMAND perl -pi -e 's/__std____array__/std::array/g' ${MODULE}.decl.h
      COMMAND perl -pi -e 's/__std____array__/std::array/g' ${MODULE}.def.h
      )
  add_custom_target(
      module_${MODULE}
      DEPENDS ${MODULE}.decl.h ${MODULE}.def.h
  )
endfunction()
