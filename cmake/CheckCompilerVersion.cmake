# Distributed under the MIT License.
# See LICENSE.txt for details.

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
    message(FATAL_ERROR "GCC version must be at least 9.1")
  endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
    message(FATAL_ERROR "Clang version must be at least 10.0")
  endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  message(FATAL_ERROR "Intel compiler is not supported.")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0.0)
    message(FATAL_ERROR "AppleClang version must be at least 11.0.0")
  endif ()
else ()
  message(
      WARNING "The compiler ${CMAKE_CXX_COMPILER_ID} is unsupported! "
      "Compilation has only been tested with Clang and GCC."
  )
endif ()
