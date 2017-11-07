# Distributed under the MIT License.
# See LICENSE.txt for details.

# Checks whether linking supports the given flag.
# Sets FLAG_WORKS to true if linking does not issue a diagnostic message
# when given FLAG_TO_CHECK
macro(CHECK_CXX_LINKER_FLAG FLAG_TO_CHECK FLAG_WORKS)
  set(SOURCE_TO_TRY "int main() { return 0; }")
  set(FILE_TO_WRITE
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx")
  file(WRITE "${FILE_TO_WRITE}" "${SOURCE_TO_TRY}\n")
  set(LINK_COMMAND_TO_TRY
    "${CHARM_COMPILER} ${FLAG_TO_CHECK} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

  if(NOT CMAKE_REQUIRED_QUIET)
    message(STATUS "Performing Test ${FLAG_TO_CHECK}")
  endif()

  try_compile(${FLAG_WORKS}
    ${CMAKE_BINARY_DIR}
    ${FILE_TO_WRITE}
    CMAKE_FLAGS -DCMAKE_CXX_LINK_EXECUTABLE:STRING=${LINK_COMMAND_TO_TRY}
    OUTPUT_VARIABLE OUTPUT)

  if(${FLAG_WORKS})
    set(${FLAG_WORKS} 1 CACHE INTERNAL "Test ${FLAG_WORKS}")
    if(NOT CMAKE_REQUIRED_QUIET)
      message(STATUS "Performing Test ${FLAG_WORKS} - Success")
    endif()
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Performing C++ SOURCE FILE Test ${FLAG_WORKS} succeeded with the following output:\n"
      "${OUTPUT}\n"
      "Source file was:\n${SOURCE_TO_TRY}\n")
  else()
    if(NOT CMAKE_REQUIRED_QUIET)
      message(STATUS "Performing Test ${FLAG_WORKS} - Failed")
    endif()
    set(${FLAG_WORKS} "" CACHE INTERNAL "Test ${FLAG_WORKS}")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
      "Performing C++ SOURCE FILE Test ${FLAG_WORKS} failed with the following output:\n"
      "${OUTPUT}\n"
      "Source file was:\n${SOURCE_TO_TRY}\n")
  endif()
endmacro()
