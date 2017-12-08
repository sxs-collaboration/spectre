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
  configure_file(
    ${CMAKE_SOURCE_DIR}/tools/SpectrePch.hpp
    ${CMAKE_BINARY_DIR}/SpectrePch.hpp
    )

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
    ${CMAKE_SOURCE_DIR}/src/ErrorHandling/Assert.hpp
    ${CMAKE_SOURCE_DIR}/src/Utilities/Blaze.hpp
    ${CMAKE_SOURCE_DIR}/src/Utilities/PointerVector.hpp
    ${CMAKE_SOURCE_DIR}/src/Utilities/TMPL.hpp
    ${PCH_PATH}
    )

  add_custom_target(
    pch
    DEPENDS
    ${PCH_PATH}
    ${PCH_PATH}.gch
    )

  # Prepend the compiler-dependent flags needed to use precompiled headers
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-include ${PCH_PATH} ${CMAKE_CXX_FLAGS}")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-include-pch ${PCH_PATH}.gch ${CMAKE_CXX_FLAGS}")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "-include-pch ${PCH_PATH}.gch ${CMAKE_CXX_FLAGS}")
  else()
    message(
      STATUS "Precompiled headers have not been configured for"
      " the compiler: ${CMAKE_CXX_COMPILER_ID}"
      )
  endif()

  # Override the default add_library and add_executable function provided
  # by CMake so that it adds the precompiled header as a dependency to
  # all of them.
  #
  # In addition to the library/executable depending on the precompiled header,
  # all source files (technically the objects generated from them) must also
  # depend on the precompiled header.
  function(add_library TARGET_NAME)
    _add_library(${TARGET_NAME} ${ARGN})
    add_dependencies(${TARGET_NAME} pch)
    set_source_files_properties(
      ${ARGN}
      OBJECT_DEPENDS "${PCH_PATH};${PCH_PATH}.gch"
      )
  endfunction()

  function(add_executable TARGET_NAME)
    _add_executable(${TARGET_NAME} ${ARGN})
    add_dependencies(${TARGET_NAME} pch)
    set_source_files_properties(
      ${ARGN}
      OBJECT_DEPENDS "${PCH_PATH};${PCH_PATH}.gch"
      )
  endfunction()

endif (USE_PCH)
