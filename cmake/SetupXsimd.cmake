# Distributed under the MIT License.
# See LICENSE.txt for details.

option(USE_XSIMD "Use xsimd if it is available" OFF)

if(USE_XSIMD)
  find_package(xsimd REQUIRED)

  if(xsimd_VERSION VERSION_LESS 11.0.1)
    message(FATAL_ERROR "xsimd must be at least 11.0.1, got ${xsimd_VERSION}")
  endif()

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
  #
  # To enable use with Blaze, add:
  #   -DBLAZE_USE_XSIMD=1
  set_property(TARGET xsimd
    APPEND PROPERTY
    INTERFACE_COMPILE_OPTIONS
    -DSPECTRE_USE_XSIMD
    )

  set_property(
    GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
    xsimd
    )

  # Currently we still have some compatibility bugs to sort out between Blaze
  # and xsimd. Once that's done we will enable combining the two.
  #
  # target_link_libraries(
  #   Blaze
  #   INTERFACE
  #   xsimd
  #   )
endif()
