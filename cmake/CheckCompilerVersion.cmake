# Distributed under the MIT License.
# See LICENSE.txt for details.

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "GCC version must be at least 7.0")
  endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
    message(FATAL_ERROR "Clang version must be at least 8.0")
  endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  message(FATAL_ERROR "Intel compiler is not supported.")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0.0)
    message(FATAL_ERROR "AppleClang version must be at least 11.0.0")
  endif ()
else ()
  message(
      WARNING "The compiler ${CMAKE_CXX_COMPILER_ID} is unsupported compiler! "
      "Compilation has only been tested with Clang, and GCC."
  )
endif ()
