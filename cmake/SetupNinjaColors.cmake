# Distributed under the MIT License.
# See LICENSE.txt for details.

if ("${CMAKE_GENERATOR}" STREQUAL "Ninja")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set_property(TARGET SpectreFlags
      APPEND PROPERTY
      INTERFACE_COMPILE_OPTIONS
      $<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:
      -fdiagnostics-color=always>)
  elseif (
      "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR
      "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set_property(TARGET SpectreFlags
      APPEND PROPERTY
      INTERFACE_COMPILE_OPTIONS
      $<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:
      -fcolor-diagnostics>)
  else ()
    message(
      WARNING "Not sure how to get color output with"
      " Ninja and compiler id ${CMAKE_CXX_COMPILER_ID}")
  endif ()
endif ()
