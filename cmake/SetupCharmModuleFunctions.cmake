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

    COMMAND perl -i -p00e 's/typename[\\s]+\([A-Za-z]\)\(?=[^\(]*\\\)\)/__typename__\\1/g' ${MODULE}.ci

    COMMAND perl -pi -e 's/std::array/__std____array__/g' ${MODULE}.ci

    COMMAND ${CHARM_COMPILER} -no-charmrun ${MODULE}.ci

    COMMAND perl -pi -e 's/__typename__/typename /g' ${MODULE}.decl.h
    COMMAND perl -pi -e 's/__typename__/typename /g' ${MODULE}.def.h
    COMMAND perl -pi -e 's/__std____array__/std::array/g' ${MODULE}.decl.h
    COMMAND perl -pi -e 's/__std____array__/std::array/g' ${MODULE}.def.h

    # Allow non-copyable types be passed to constructors of ConstGlobalCache
    COMMAND perl -pi -e 's/\(new \\\(impl_obj_void\\\)\\s*${MODULE}\\s*<\\s*Metavariables\\s*>\\s*\\\(\)impl_noname_0/\\1std::move\(impl_noname_0\)/g' ${MODULE}.def.h
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
  #
  # Because multiple COMMANDs in a single execute_process statement results in
  # piping of output from one command to another, we just do an && between all
  # the different calls. Separating into separate execute_process calls is
  # slow and bloats the code a lot. For documentation we segment the code into
  # blocks, each block is sectioned off by blank lines
  #
  # First block:
  # Move into user class from charm for receive_data, allowing perfect
  # forwarding
  #
  # Second block:
  # Handle entry methods that take a variadic std::tuple. Charm++ cannot
  # handle variadic entry methods, so we must hack this in a little
  #
  # Third block:
  # Use move semantics in simple_action
  execute_process(
    COMMAND bash -c
    "perl -pi -e 's@\(\\s*impl_obj->template\\s+receive_data<\\s*ReceiveTag\),\\s*ReceiveData_t\\s*>\\\(\(impl_noname_\\d\),\\s*\(impl_noname_\\d\)@\\1>\(std::move\(\\2\), std::move\(\\3\)@g' Algorithm${ALGORITHM_NAME}.def.h \
     \
     \
     && perl -pi -e 's/LDOTLDOTLDOT/.../g' Algorithm${ALGORITHM_NAME}.def.h \
     && perl -pi -e 's/LDOTLDOTLDOT/.../g' Algorithm${ALGORITHM_NAME}.decl.h \
     && perl -pi -e 's/\(\\\"simple_action\\\(const\\s+std::tuple<\\s*\)COMPUTE_VARIADIC_ARGS\(\\s*>\\s*&args\\\)\\\"\)/std::string\(std::string\(\\1\"\) + Parallel::charmxx::get_template_parameters_as_string<Args...>\(\) + std::string\(\"\\2\)\).c_str\(\)/g' Algorithm${ALGORITHM_NAME}.def.h \
     && perl -pi -e 's/COMPUTE_VARIADIC_ARGS/Args.../g' \
             Algorithm${ALGORITHM_NAME}.def.h \
     && perl -pi -e 's/COMPUTE_VARIADIC_ARGS/Args.../g' \
             Algorithm${ALGORITHM_NAME}.decl.h \
     && perl -pi -e 's/<\\s*Action, Args\\s*>/<Action, Args...>/g' \
             Algorithm${ALGORITHM_NAME}.def.h \
     && perl -pi -e 's/<\\s*Action, Args\\s*>/<Action, Args...>/g' \
             Algorithm${ALGORITHM_NAME}.decl.h \
     && perl -pi -e 's/METAVARIABLES_FROM_COMPONENT/typename ParallelComponent::metavariables/g' \
             Algorithm${ALGORITHM_NAME}.def.h \
     && perl -pi -e 's/METAVARIABLES_FROM_COMPONENT/typename ParallelComponent::metavariables/g' \
             Algorithm${ALGORITHM_NAME}.decl.h \
     \
     \
     && perl -pi -e 's/\(\\s*impl_obj->template\\s+simple_action<\\s*Action\)\\s*>\\\(args/\\1>\(std::move\(args\)/g' Algorithm${ALGORITHM_NAME}.def.h \
     && perl -pi -e 's/\(\\s*impl_obj->template\\s+simple_action<\\s*Action, Args...\)\\s*>\\\(args/\\1>\(std::move\(args\)/g' Algorithm${ALGORITHM_NAME}.def.h \
     \
     \
     && perl -pi -e 's/ReceiveTag_temporal_id/typename ReceiveTag::temporal_id/g' Algorithm${ALGORITHM_NAME}.def.h \
     && perl -pi -e 's/ReceiveTag_temporal_id/typename ReceiveTag::temporal_id/g' Algorithm${ALGORITHM_NAME}.decl.h"

    WORKING_DIRECTORY ${ALGORITHM_DIR}
    RESULT_VARIABLE SUCCEEDED_GENERATING_CHARM
    )

  if (NOT ${SUCCEEDED_GENERATING_CHARM} EQUAL 0)
    message(FATAL_ERROR
      "Failed general decl.h and def.h patches for ${ALGORITHM_NAME}")
  endif()

  if ("${ALGORITHM_TYPE}" STREQUAL "array")
    execute_process(
      # First block:
      # Have receive_data use perfect forwarding
      #
      # Second block:
      # Fix inline entry method receive_data to correctly call a template
      # function, and use perfect forwarding
      #
      # Third block:
      # Add handling of custom array indices by replacing
      # CkArrayIndexSpectreArrayIndex
      #   ->     Parallel::charmxx::ArrayIndex<SpectreArrayIndex>
      COMMAND bash -c
      "perl -pi -e 's/void receive_data\\\(const typename ReceiveTag::temporal_id &impl_noname_1, const ReceiveData_t &impl_noname_2, bool enable_if_disabled = false,/void receive_data\(typename ReceiveTag::temporal_id impl_noname_1, ReceiveData_t &&impl_noname_2, bool enable_if_disabled = false,/g' Algorithm${ALGORITHM_NAME}.decl.h \
       && perl -pi -e 's/::receive_data\\\(const typename ReceiveTag::temporal_id &impl_noname_1, const ReceiveData_t &impl_noname_2, bool enable_if_disabled,/::receive_data\(typename ReceiveTag::temporal_id impl_noname_1, ReceiveData_t &&impl_noname_2, bool enable_if_disabled,/g' Algorithm${ALGORITHM_NAME}.def.h \
       \
       \
       && perl -pi -e 's/obj->receive_data\\\(impl_noname_1, impl_noname_2/obj->template receive_data<ReceiveTag>\(std::move\(impl_noname_1\), std::forward<ReceiveData_t>\(impl_noname_2\)/g' Algorithm${ALGORITHM_NAME}.def.h \
       \
       \
       && perl -pi -e 's/CkArrayIndexSpectreArrayIndex/Parallel::ArrayIndex<SpectreArrayIndex>/g' Algorithm${ALGORITHM_NAME}.decl.h"

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
