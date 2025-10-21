# Distributed under the MIT License.
# See LICENSE.txt for details.

# To turn this off, one should provide an alternative
# SIMD backend that supports elementary functions
# or the auto-differentiation in SpECTRE ceases to
# work
option(USE_XSIMD "Use xsimd if it is available" ON)

if(USE_XSIMD)
  find_package(xsimd QUIET)

  if (NOT xsimd_FOUND)
    if (NOT SPECTRE_FETCH_MISSING_DEPS)
      message(FATAL_ERROR "Could not find xsimd. If you want to fetch "
        "missing dependencies automatically, set SPECTRE_FETCH_MISSING_DEPS=ON.")
    endif()

    message(STATUS "Fetching xsimd")
    include(FetchContent)
    FetchContent_Declare(xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd
        GIT_TAG 13.2.0
        ${SPECTRE_FETCHCONTENT_BASE_ARGS}
    )
    FetchContent_MakeAvailable(xsimd)
    set(xsimd_INCLUDE_DIRS ${xsimd_SOURCE_DIR}/include)
    set(xsimd_VERSION "13.2.0")
    if (NOT CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
      get_target_property(XSIMD_IID xsimd INTERFACE_INCLUDE_DIRECTORIES)
      set_target_properties(xsimd PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${XSIMD_IID}")
    endif()
  endif()

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
  target_compile_definitions(
    xsimd
    INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:SPECTRE_USE_XSIMD>
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
