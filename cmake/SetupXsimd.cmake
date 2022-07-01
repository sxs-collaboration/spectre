# Distributed under the MIT License.
# See LICENSE.txt for details.

# QUIET silences warning
find_package(xsimd 8.1.0 QUIET)

option(USE_XSIMD "Use xsimd if it is available" ON)

if(USE_XSIMD AND xsimd_FOUND)
  message(STATUS "xsimd incld: ${xsimd_INCLUDE_DIRS}")
  message(STATUS "xsimd vers: ${xsimd_VERSION}")

  file(APPEND
    "${CMAKE_BINARY_DIR}/BuildInfo.txt"
    "xsimd version: ${xsimd_VERSION}\n"
    )

  add_interface_lib_headers(
    TARGET xsimd
    HEADERS
    xsimd/xsimd.hpp
    )

  # As long as we want xsimd support to be optional we need to be
  # able to figure out if we have it available.
  set_property(TARGET xsimd
    APPEND PROPERTY
    INTERFACE_COMPILE_OPTIONS
    -DSPECTRE_USE_XSIMD
    )

  set_property(
    GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
    xsimd
    )
endif()
