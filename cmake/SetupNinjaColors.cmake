# Distributed under the MIT License.
# See LICENSE.txt for details.

if ("${CMAKE_GENERATOR}" STREQUAL "Ninja")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-fdiagnostics-color=always ${CMAKE_C_FLAGS}")
  elseif (
      "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR
      "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "-fcolor-diagnostics ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-fcolor-diagnostics ${CMAKE_C_FLAGS}")
  else ()
    message(
      WARNING "Not sure how to get color output with"
      " Ninja and compiler id ${CMAKE_CXX_COMPILER_ID}")
  endif ()
endif ()
