# Distributed under the MIT License.
# See LICENSE.txt for details.

option(
  USE_PCH
  "Use precompiled header file tools/SpectrePch.hpp"
  ON
  )

if (USE_PCH)
  # CMake does not have any support for precompiled headers (PCHs) before
  # v3.16. After v3.16 PCHs are supported, but targets cannot share a PCH,
  # which is exactly what we want to do. To this end, we implement handling
  # of a global PCH (and could be generalized to multiple PCHs) below.
  # Since the setup is somewhat complicated we give an overview of what is
  # happening here, and various things to be aware of.
  #
  # We have the following requirements on the PCH:
  # - it must be rebuilt whenever anything it includes is changed.
  # - it must be removed when running `make clean`
  # - it must be generated before any object files are built that
  #   use it
  # - it must be able to use flags from other targets (since in modern CMake
  #   flags, etc. are propagated via targets)
  # - the correct flags to include the PCH must be propagated to any targets
  #   that use the PCH.
  #
  # We store the header that generates the PCH in ${CMAKE_SOURCE_DIR}/tools
  # (variable ${SPECTRE_PCH_HEADER_SOURCE_PATH}) so that it is not
  # accidentally included anywhere. We then symlink the header used to
  # generate PCH to ${SPECTRE_PCH_HEADER_PATH}. The PCH will be generated
  # as ${SPECTRE_PCH_PATH}.
  #
  # In order to track the PCH include dependencies we symlink
  # ${SPECTRE_PCH_HEADER_SOURCE_PATH} into the build dir as
  # ${SPECTRE_PCH_DEP_HEADER_PATH} and generate a source file
  # ${SPECTRE_PCH_DEP_SOURCE_PATH} that includes ${SPECTRE_PCH_DEP_HEADER_PATH}
  # and thus tracks the includes of its dependencies. If we were to include
  # ${SPECTRE_PCH_HEADER_PATH} instead of ${SPECTRE_PCH_DEP_HEADER_PATH} then
  # the compiler will automatically grab the PCH instead of doing a textual
  # include, which breaks the dependency tracking. A library,
  # ${SPECTRE_PCH_LIB} is created that builds
  # ${SPECTRE_PCH_DEP_SOURCE_PATH}, but using `tools/WrapPchCompiler.sh`
  # (achieved by setting the RULE_LAUNCH_COMPILE property on
  # ${SPECTRE_PCH_LIB}).
  #
  # We wrap the compiler so that we can build both the object file and
  # the PCH with the full flags. That is, it is pretty much impossible to
  # figure out all the compiler flags (especially include flags) and so using
  # a CMake custom command to invoke the compiler is really difficult (more
  # on this below). The compiler wrapper script is very simple, it invokes the
  # commands passed into it to build the object file, as well as strips out
  # `-o XX -c YY` and then replaces them with
  # `${SPECTRE_PCH_HEADER_PATH} -o ${SPECTRE_PCH_PATH}`. We make the
  # ${SPECTRE_PCH_SOURCE_PATH}'s resulting object file depend on
  # ${SPECTRE_PCH_COMPILER_WRAPPER} so that if the wrapper is changed the PCH
  # is rebuilt (source file property OBJECT_DEPENDS). We also tell CMake that
  # the source file build outputs ${SPECTRE_PCH_PATH} by setting the
  # OBJECT_OUTPUTS property on ${SPECTRE_PCH_SOURCE_PATH}.
  #
  # To get CMake to track the PCH file rather than the object file we create
  # the custom target ${SPECTRE_PCH} and have it depend on ${SPECTRE_PCH_LIB}.
  # With the Ninja generator we can directly depend on the PCH file, but not
  # with Makefiles. Any library or executable that should include the PCH
  # must now depend on ${SPECTRE_PCH}.
  #
  # The next issue to deal with is having the PCH be removed by `make clean`
  # or `ninja clean` calls (depending on which generator was used). Before
  # CMake v3.15 the variable ${ADDITIONAL_MAKE_CLEAN_FILES} can be used to
  # have `make clean` remove files, but this does not work when
  # using Ninja. From CMake V3.15 onward the variable ${ADDITIONAL_CLEAN_FILES}
  # can be used to specify additional files to remove regardless of the
  # generator used.
  #
  # The major remaining step is setting the PCH include flag. For
  # (Apple)Clang this is `-include-pch;${SPECTRE_PCH_PATH}`, while for GCC it
  # is `-include;${SPECTRE_PCH_HEADER_PATH}`. We add
  # `INTERFACE_COMPILE_OPTIONS` to ${SPECTRE_PCH} that libraries can grab
  # using generator expressions:
  #
  #     target_compile_options(
  #       ${TARGET_NAME}
  #       PRIVATE
  #       $<TARGET_PROPERTY:${SPECTRE_PCH},INTERFACE_COMPILE_OPTIONS>
  #       )
  #
  # Source files in dependent targets must depend on the PCH file itself:
  #
  #     set_source_files_properties(
  #       ${ARGN}
  #       OBJECT_DEPENDS "${SPECTRE_PCH_PATH}"
  #       )
  #
  # Finally, the add_spectre_library and add_spectre_executable functions
  # both will have targets automatically include the PCH.
  #
  # Additional notes:
  # - We used to have a custom_command that would invoke the compiler, which
  #   required us to filter out compiler-internal include directories
  #   (e.g. /usr/include) so as not to -isystem them and break compilation
  #   for people who have Blaze, Brigand, or any other dependencies of the PCH
  #   somewhere like `/usr/include`. Unfortunately, determining what to filter
  #   out is difficult. CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES used to work, but
  #   around CMake v3.11 CMake started adding the entries in CPATH into that,
  #   which resulted in the PCH spewing warnings for anyone who didn't have
  #   Blaze in a compiler-internal include directory (e.g. /usr/include).
  #   CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES is also not helpful, and so the
  #   only way to continue with this design would be to create a copy of
  #   CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES, then remove everything in
  #   $ENV{CPATH} from the copy, and avoid adding `-isystem` for any include
  #   directories in the list. This design still doesn't allow for a good
  #   way of handling flags that are passed around as targets, which is
  #   the modern CMake way of handling flags rather than modifying
  #   ${CMAKE_CXX_FLAGS}.

  # Set up variables for PCH.
  set(SPECTRE_PCH SpectrePch)
  set(SPECTRE_PCH_HEADER_SOURCE_PATH
    "${CMAKE_SOURCE_DIR}/tools/SpectrePch.hpp")
  set(SPECTRE_PCH_HEADER_PATH "${CMAKE_BINARY_DIR}/SpectrePch.hpp")
  set(SPECTRE_PCH_PATH "${SPECTRE_PCH_HEADER_PATH}.gch")
  set(SPECTRE_PCH_LIB SpectrePchLib)
  set(SPECTRE_PCH_LIB_DIR "${CMAKE_BINARY_DIR}/tmp/")
  set(SPECTRE_PCH_DEP_HEADER_PATH
    "${SPECTRE_PCH_LIB_DIR}.SpectrePchForDependencies.hpp")
  set(SPECTRE_PCH_DEP_SOURCE_PATH
    "${SPECTRE_PCH_LIB_DIR}.SpectrePchForDependencies.cpp")
  set(SPECTRE_PCH_COMPILER_WRAPPER
    "${CMAKE_BINARY_DIR}/tmp/WrapPchCompiler.sh")

  # Symlink header file for PCH
  execute_process(
    COMMAND ${CMAKE_COMMAND}
    -E create_symlink
    ${SPECTRE_PCH_HEADER_SOURCE_PATH}
    ${SPECTRE_PCH_HEADER_PATH}
    )

  # Generate SPECTRE_PCH_LIB_DIR, and symlink SPECTRE_PCH_DEP_HEADER_PATH
  if(NOT EXISTS ${SPECTRE_PCH_LIB_DIR})
    file(MAKE_DIRECTORY ${SPECTRE_PCH_LIB_DIR})
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND}
    -E create_symlink
    ${SPECTRE_PCH_HEADER_SOURCE_PATH}
    ${SPECTRE_PCH_DEP_HEADER_PATH}
    RESULT_VARIABLE RESULT_OF_LINK)
  if(NOT ${RESULT_OF_LINK} EQUAL 0)
    message(FATAL_ERROR
      "Failed to create symbolic link for ${SPECTRE_PCH_DEP_HEADER_PATH}")
  endif()

  # Set up compiler wrapper
  configure_file(
    ${CMAKE_SOURCE_DIR}/tools/WrapPchCompiler.sh
    ${SPECTRE_PCH_COMPILER_WRAPPER}
    @ONLY
    )

  # We write a temp file and use configure_file so we don't trigger
  # a rebuild of the PCH every time CMake is run.
  file(WRITE
    ${SPECTRE_PCH_DEP_SOURCE_PATH}.out
    "#include \"${SPECTRE_PCH_DEP_HEADER_PATH}\"\n"
    )
  configure_file(
    ${SPECTRE_PCH_DEP_SOURCE_PATH}.out
    ${SPECTRE_PCH_DEP_SOURCE_PATH}
    )

  # Create SPECTRE_PCH_LIB, specify ${SPECTRE_PCH_LIB_DIR} and
  # ${SPECTRE_PCH_COMPILER_WRAPPER}
  add_library(
    ${SPECTRE_PCH_LIB}
    ${SPECTRE_PCH_DEP_SOURCE_PATH}
    )
  target_link_libraries(
    ${SPECTRE_PCH_LIB}
    PRIVATE
    Blaze
    Brigand
    )
  target_include_directories(
    ${SPECTRE_PCH_LIB}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    )
  set_target_properties(
    ${SPECTRE_PCH_LIB}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${SPECTRE_PCH_LIB_DIR}
    RULE_LAUNCH_COMPILE ${SPECTRE_PCH_COMPILER_WRAPPER}
    DEFINE_SYMBOL ""
    )
  target_link_libraries(
    ${SPECTRE_PCH_LIB}
    PRIVATE
    SpectreFlags
    )

  # Make the source file ${SPECTRE_PCH_DEP_SOURCE_PATH} depend on
  # ${SPECTRE_PCH_COMPILER_WRAPPER} and set the source file to produce
  # ${SPECTRE_PCH_PATH}.
  set_source_files_properties(
    ${SPECTRE_PCH_DEP_SOURCE_PATH}
    PROPERTIES
    OBJECT_DEPENDS ${SPECTRE_PCH_COMPILER_WRAPPER}
    OBJECT_OUTPUTS ${SPECTRE_PCH_PATH}
    )

  # Create the ${SPECTRE_PCH} that libraries and executables will depend on
  add_custom_target(
    ${SPECTRE_PCH}
    )
  add_dependencies(
    ${SPECTRE_PCH}
    ${SPECTRE_PCH_LIB}
    )

  # Set the compiler-dependent flags needed to use precompiled headers
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(SPECTRE_PCH_FLAG "-include;${SPECTRE_PCH_HEADER_PATH}")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(SPECTRE_PCH_FLAG "-include-pch;${SPECTRE_PCH_PATH}")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(SPECTRE_PCH_FLAG "-include-pch;${SPECTRE_PCH_PATH}")
  else()
    message(
      STATUS "Precompiled headers have not been configured for"
      " the compiler: ${CMAKE_CXX_COMPILER_ID}"
      )
  endif()

  set_property(TARGET ${SPECTRE_PCH}
    APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CXX>:${SPECTRE_PCH_FLAG}>
    )

  # Have `make clean` (and `ninja clean` on CMake v3.15 and newer) remove
  # the PCH.
  if(CMAKE_VERSION VERSION_EQUAL 3.15 OR CMAKE_VERSION VERSION_GREATER 3.15)
    set_property(DIRECTORY APPEND PROPERTY
      ADDITIONAL_CLEAN_FILES "${SPECTRE_PCH_PATH}")
  else(CMAKE_VERSION VERSION_EQUAL 3.15 OR CMAKE_VERSION VERSION_GREATER 3.15)
    set_property(DIRECTORY APPEND PROPERTY
      ADDITIONAL_MAKE_CLEAN_FILES "${SPECTRE_PCH_PATH}")
  endif(CMAKE_VERSION VERSION_EQUAL 3.15 OR CMAKE_VERSION VERSION_GREATER 3.15)
else (USE_PCH)
  set(SPECTRE_PCH "")
endif (USE_PCH)
