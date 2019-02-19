# Distributed under the MIT License.
# See LICENSE.txt for details.

option(
  USE_PCH
  "Use precompiled headers for STL, Blaze, and Brigand"
  ON
  )

function(is_implicit_include_directory INCLUDE_DIR)
  set(FOUND_IN_IMPLICIT_INCLUDE_DIRECTORIES OFF PARENT_SCOPE)
  foreach(DIR ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
    if("${INCLUDE_DIR}" STREQUAL "${DIR}")
      set(FOUND_IN_IMPLICIT_INCLUDE_DIRECTORIES ON PARENT_SCOPE)
    endif()
  endforeach()
endfunction()

if (USE_PCH)
  # We store the header to precompile in ${CMAKE_SOURCE_DIR}/tools so
  # that it cannot accidentally be included incorrectly anywhere since
  # ${CMAKE_SOURCE_DIR}/tools is not in the include list.
  # We copy it to the build directory and include it from there because
  # GCC's precompiled headers require the hpp and the precompiled header
  # be in the same directory.
  set(PCH_PATH "${CMAKE_BINARY_DIR}/SpectrePch.hpp")
  execute_process(
    COMMAND ${CMAKE_COMMAND}
    -E create_symlink
    ${CMAKE_SOURCE_DIR}/tools/SpectrePch.hpp
    ${CMAKE_BINARY_DIR}/SpectrePch.hpp
    )
  # We create a second copy of the PCH and also a simple source file that
  # includes the second copy, which we compile into an unused library. The
  # library will be a dependency of the real PCH so that way if anything
  # changes that the PCH depends on the PCH will be updated. While this means
  # we technically compile the PCH twice, it is a generator-independent way of
  # handling the dependencies. The other methods available in CMake 3.3 are
  # generator dependent, even all the ones in CMake 3.11 are. We always add
  # the PCH library as a static lib so we know the generated libs name.
  set(PCH_LIB_NAME "PCH_SPECTRE_DEPENDENCIES")
  set(PCH_LIB_DIR "${CMAKE_BINARY_DIR}/tmp/")
  if(NOT EXISTS "${CMAKE_BINARY_DIR}/tmp/")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tmp/")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND}
    -E create_symlink
    ${CMAKE_SOURCE_DIR}/tools/SpectrePch.hpp
    ${CMAKE_BINARY_DIR}/tmp/.SpectrePchForDependencies.hpp
    RESULT_VARIABLE RESULT_OF_LINK)
  if(NOT ${RESULT_VARIABLE} EQUAL 0)
    message(FATAL_ERROR
      "Failed to create symbolic link for "
      "${CMAKE_BINARY_DIR}/tmp/.SpectrePchForDependencies.hpp")
  endif()
  # We write a temp file and use configure_file so we don't trigger
  # a rebuild of the PCH every time CMake is run.
  file(WRITE
    ${CMAKE_BINARY_DIR}/tmp/.SpectrePchForDependencies.cpp.out
    "#include \"${CMAKE_BINARY_DIR}/tmp/.SpectrePchForDependencies.hpp\"\n"
    )
  configure_file(
    ${CMAKE_BINARY_DIR}/tmp/.SpectrePchForDependencies.cpp.out
    ${CMAKE_BINARY_DIR}/tmp/.SpectrePchForDependencies.cpp
    )
  add_library(
    ${PCH_LIB_NAME} STATIC
    ${CMAKE_BINARY_DIR}/tmp/.SpectrePchForDependencies.cpp
    )
  set_target_properties(
    PCH_SPECTRE_DEPENDENCIES
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${PCH_LIB_DIR}
    )
  set(PCH_LIB_PATH "${PCH_LIB_DIR}/lib${PCH_LIB_NAME}.a")

  # The compiler flags need to be turned into a CMake list
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    string(REPLACE " " ";" PCH_COMPILE_FLAGS
      "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}"
      )
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "None")
    string(REPLACE " " ";" PCH_COMPILE_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    string(REPLACE " " ";" PCH_COMPILE_FLAGS
      "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}"
      )
  endif()

  is_implicit_include_directory(${BLAZE_INCLUDE_DIR})
  if(NOT ${FOUND_IN_IMPLICIT_INCLUDE_DIRECTORIES})
    set(BLAZE_INCLUDE_ARGUMENT "-isystem${BLAZE_INCLUDE_DIR}")
  else()
    set(BLAZE_INCLUDE_ARGUMENT "")
  endif()

  is_implicit_include_directory(${BRIGAND_INCLUDE_DIR})
  if(NOT ${FOUND_IN_IMPLICIT_INCLUDE_DIRECTORIES})
    set(BRIGAND_INCLUDE_ARGUMENT "-isystem${BRIGAND_INCLUDE_DIR}")
  else()
    set(BRIGAND_INCLUDE_ARGUMENT "")
  endif()

  is_implicit_include_directory(${CHARM_INCLUDE_DIRS})
  if(NOT ${FOUND_IN_IMPLICIT_INCLUDE_DIRECTORIES})
    set(CHARM_INCLUDE_ARGUMENT "-isystem${CHARM_INCLUDE_DIRS}")
  else()
    set(CHARM_INCLUDE_ARGUMENT "")
  endif()

  add_custom_command(
    OUTPUT ${PCH_PATH}.gch
    COMMAND ${CMAKE_CXX_COMPILER}
    ARGS
    -std=c++14
    ${PCH_COMPILE_FLAGS}
    -I${CMAKE_SOURCE_DIR}/src
    ${BLAZE_INCLUDE_ARGUMENT}
    ${BRIGAND_INCLUDE_ARGUMENT}
    ${CHARM_INCLUDE_ARGUMENT}
    ${PCH_PATH}
    -o ${PCH_PATH}.gch
    DEPENDS
    ${PCH_LIB_PATH}
    )

  add_custom_target(
    pch
    DEPENDS
    ${PCH_PATH}.gch
    ${PCH_LIB_PATH}
    )

  add_dependencies(
    pch
    ${PCH_LIB_NAME}
    )

  # Prepend the compiler-dependent flags needed to use precompiled headers
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(PCH_FLAG "-include;${PCH_PATH}")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(PCH_FLAG "-include-pch;${PCH_PATH}.gch")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(PCH_FLAG "-include-pch;${PCH_PATH}.gch")
  else()
    message(
      STATUS "Precompiled headers have not been configured for"
      " the compiler: ${CMAKE_CXX_COMPILER_ID}"
      )
  endif()

  # Override the default add_library function provided by CMake and override
  # add_spectre_executable function so that it adds the precompiled header as a
  # dependency to all of them.
  #
  # In addition to the library/executable depending on the precompiled header,
  # all source files (technically the objects generated from them) must also
  # depend on the precompiled header.
  function(add_library TARGET_NAME)
    _add_library(${TARGET_NAME} ${ARGN})
    get_target_property(
      TARGET_IS_IMPORTED
      ${TARGET_NAME}
      IMPORTED
      )
    if (NOT "${TARGET_NAME}" MATCHES "^PCH"
        AND NOT ${TARGET_IS_IMPORTED})
      add_dependencies(${TARGET_NAME} pch)
      set_source_files_properties(
        ${ARGN}
        OBJECT_DEPENDS "${PCH_PATH};${PCH_PATH}.gch"
        )
      target_compile_options(
        ${TARGET_NAME}
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${PCH_FLAG}>
        )
    endif()
  endfunction()
endif (USE_PCH)
