# Distributed under the MIT License.
# See LICENSE.txt for details.

# Use CCache if available to speed up builds
#
# CCache is particularly useful for speeding up builds on CI, where we can share
# the cache between runs. It is also very useful for general development.
#
# To support precompiled headers (PCH) we follow the recommendations in the
# ccache docs:
# https://ccache.dev/manual/latest.html#_precompiled_headers
# Our requirements for CCache with PCH are:
# - Works with GCC and Clang.
# - Deleting and recreating the build directory retains ~100% cache hit rate.
# - `make clean` retains ~100% cache hit rate.
# - Changing the PCH file in tools/SpectrePch.hpp invalidates the cache.
# - Sharing caches between runs on CI works.

option(USE_CCACHE "Use CCache if available to speed up builds" ON)
if(USE_CCACHE)
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    string(CONCAT CCACHE_COMMAND
      "CCACHE_SLOPPINESS=pch_defines,time_macros,"
      "include_file_mtime,include_file_ctime "
      "ccache "
    )
    # CCache offers no benefit for linking
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_COMMAND})
    message(STATUS "Using ccache for compilation. It is invoked as: ${CCACHE_COMMAND}")
  else()
    message(STATUS "Could not find ccache")
  endif(CCACHE_FOUND)
endif(USE_CCACHE)

# Add `-fno-pch-timestamp` flag to Clang to support precompiled headers
# (see https://ccache.dev/manual/4.8.2.html#_precompiled_headers)
# Note that `CMAKE_CXX_COMPILE_OPTIONS_CREATE_PCH` isn't part of the public API
# so this may break in the future. For a discussion see:
# https://discourse.cmake.org/t/ccache-clang-and-fno-pch-timestamp/7253/6
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND CMAKE_CXX_COMPILE_OPTIONS_CREATE_PCH -Xclang -fno-pch-timestamp)
endif()
