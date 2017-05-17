
# Distributed under the MIT License.
# See LICENSE.txt for details.

set(CMAKE_BUILD_TYPES "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE not specified, setting to 'Debug'")
  set(
      CMAKE_BUILD_TYPE Debug
      CACHE STRING "Choose the type of build: ${CMAKE_BUILD_TYPES}"
      FORCE
  )
else()
  if(NOT ${CMAKE_BUILD_TYPE} IN_LIST CMAKE_BUILD_TYPES)
    message(
        FATAL_ERROR "\n"
        "Invalid value for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}\n"
        "Valid values: ${CMAKE_BUILD_TYPES}\n"
    )
  endif()
  message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()
