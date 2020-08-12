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

function(generate_algorithms_impl ALGORITHM_NAME ALGORITHM_TYPE ALGORITHM_DIR)
  # Arguments:
  #  ALGORITHM_NAME: the name of the algorithm that will be post-fixed
  #                  to "Algorithm"
  #  ALGORITHM_TYPE: one of:
  #                    chare - A singleton
  #                    array - A chare array
  #                    group - A group, which is one chare per core
  #                    nodegroup - A nodegroup, one chare per "node"
  #  ALGORITHM_DIR: the output dir where the header files are generated

  string(TOLOWER ${ALGORITHM_TYPE} ALGORITHM_TYPE)
  execute_process(
    WORKING_DIRECTORY ${ALGORITHM_DIR}

    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/cmake/SetupCharmAlgorithm.py
    --algorithm_name "${ALGORITHM_NAME}"
    --algorithm_type "${ALGORITHM_TYPE}"
    --build_dir "${ALGORITHM_DIR}"

    RESULT_VARIABLE SUCCEEDED_GENERATING_CHARM
    )

  if (${SUCCEEDED_GENERATING_CHARM} EQUAL 255)
    # The python script returns 255 if the ci and hpp files exist
    # and are up-to-date. In this case we do not need to run charmc again
    # and can save compilation time by not "updating" up-to-date files.
    return()
  endif()

  if (NOT ${SUCCEEDED_GENERATING_CHARM} EQUAL 0)
    message(FATAL_ERROR
      "Failed to run script SetupCharmAlgorithm.py for "
      "algorithm: ${ALGORITHM_NAME}")
  endif()

  # Run the charmc script over the generated ci file to generate the decl.h
  # and def.h header files
  execute_process(
    COMMAND ${CHARM_COMPILER} -no-charmrun Algorithm${ALGORITHM_NAME}.ci

    WORKING_DIRECTORY ${ALGORITHM_DIR}
    RESULT_VARIABLE SUCCEEDED_GENERATING_CHARM
    )

  if (NOT ${SUCCEEDED_GENERATING_CHARM} EQUAL 0)
    message(FATAL_ERROR
      "${CHARM_COMPILER} failed for algorithm: ${ALGORITHM_NAME}")
  endif()

  # Fix bugs in the generated decl.h and def.h files using regex
  if ("${ALGORITHM_TYPE}" STREQUAL "array")
    execute_process(
      # First block:
      # Add handling of custom array indices by replacing
      # CkArrayIndexSpectreArrayIndex
      #   ->     Parallel::charmxx::ArrayIndex<SpectreArrayIndex>
      COMMAND bash -c
      "perl -pi -e 's/CkArrayIndexSpectreArrayIndex/Parallel::ArrayIndex<SpectreArrayIndex>/g' Algorithm${ALGORITHM_NAME}.decl.h"

      WORKING_DIRECTORY ${ALGORITHM_DIR}
      RESULT_VARIABLE SUCCEEDED_GENERATING_CHARM
      )

    if (NOT ${SUCCEEDED_GENERATING_CHARM} EQUAL 0)
      message(FATAL_ERROR
        "Failed to patch generated ${ALGORITHM_TYPE} files")
    endif()
  endif()
endfunction()

function(generate_algorithms ALGORITHM_DIR)
  # Generates the 4 different types of algorithms into the
  # ALGORITHM_DIR directory. The directory is created if it
  # does not exits. ALGORITHM_DIR is also added to the include
  # directories
  if (NOT EXISTS ${ALGORITHM_DIR})
    FILE(MAKE_DIRECTORY ${ALGORITHM_DIR})
  endif()
  include_directories(${ALGORITHM_DIR})

  generate_algorithms_impl(Singleton chare ${ALGORITHM_DIR})
  generate_algorithms_impl(Array array ${ALGORITHM_DIR})
  generate_algorithms_impl(Group group ${ALGORITHM_DIR})
  generate_algorithms_impl(Nodegroup nodegroup ${ALGORITHM_DIR})
endfunction()
