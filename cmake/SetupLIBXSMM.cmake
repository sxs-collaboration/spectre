# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(LIBXSMM 1.16.1)

if (NOT LIBXSMM_FOUND)
  if (NOT SPECTRE_FETCH_MISSING_DEPS)
    message(FATAL_ERROR "Could not find LIBXSMM. If you want to fetch "
      "missing dependencies automatically, set SPECTRE_FETCH_MISSING_DEPS=ON.")
  endif()
  message(STATUS "Fetching LIBXSMM")

  # This FetchContent code is adapted from the libxsmm docs:
  # https://libxsmm.readthedocs.io/en/latest/#rules-for-building-libxsmm
  include(FetchContent)
  # Need an unreleased version for Apple Silicon chips
  if (APPLE AND ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64"
                OR "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
    FetchContent_Declare(xsmm
      GIT_REPOSITORY https://github.com/libxsmm/libxsmm.git
      GIT_TAG 939f11042fc9ae4bbe975cedb2330d4f9f4bb26e
      ${SPECTRE_FETCHCONTENT_BASE_ARGS}
    )
  else()
    FetchContent_Declare(xsmm
      URL https://github.com/libxsmm/libxsmm/archive/1.16.1.tar.gz
      ${SPECTRE_FETCHCONTENT_BASE_ARGS}
    )
  endif()
  FetchContent_GetProperties(xsmm)
  if(NOT xsmm_POPULATED)
    FetchContent_Populate(xsmm)
  endif()

  set(LIBXSMMROOT ${xsmm_SOURCE_DIR})
  file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/src/*.c)
  list(REMOVE_ITEM _GLOB_XSMM_SRCS ${LIBXSMMROOT}/src/libxsmm_generator_gemm_driver.c)
  set(XSMM_INCLUDE_DIRS ${LIBXSMMROOT}/include)

  add_library(xsmm STATIC ${_GLOB_XSMM_SRCS})
  target_include_directories(xsmm SYSTEM PUBLIC ${XSMM_INCLUDE_DIRS})
  target_compile_definitions(xsmm PUBLIC LIBXSMM_DEFAULT_CONFIG)

  # Link BLAS
  find_package(BLAS REQUIRED)
  target_link_libraries(xsmm PUBLIC BLAS::BLAS)

  # Provide `Libxsmm` target
  add_library(Libxsmm ALIAS xsmm)
endif()

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Libxsmm
  )
