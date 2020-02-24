# Distributed under the MIT License.
# See LICENSE.txt for details.

option(DEBUG_SYMBOLS "Add -g to CMAKE_CXX_FLAGS if ON, -g0 if OFF." ON)
option(SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
  "Rigorously check protocol conformance instead of deferring to unit tests. \
May significantly increase compile time."
  OFF)

option(OVERRIDE_ARCH "The architecture to use. Default is native." OFF)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSPECTRE_DEBUG")

if(NOT ${DEBUG_SYMBOLS})
  string(REPLACE "-g " "-g0 " CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif()

# Always build with -g so we can view backtraces, etc. when production code
# fails. This can be overridden by passing `-D DEBUG_SYMBOLS=OFF` to CMake
if(${DEBUG_SYMBOLS})
  set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS}")
endif(${DEBUG_SYMBOLS})

# Always compile only for the current architecture. This can be overridden
# by passing `-D OVERRIDE_ARCH=THE_ARCHITECTURE` to CMake
if(NOT "${OVERRIDE_ARCH}" STREQUAL "OFF")
  set(CMAKE_CXX_FLAGS "-march=${OVERRIDE_ARCH} ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
endif()

# We always want a detailed backtrace of template errors to make debugging them
# easier
set(CMAKE_CXX_FLAGS "-ftemplate-backtrace-limit=0 ${CMAKE_CXX_FLAGS}")

# We disable thread safety of Boost::shared_ptr since it makes them faster
# to use and we do not share them between threads. If a thread-safe
# shared_ptr is desired it must be implemented to work with Charm++'s threads
# anyway.
add_definitions(-DBOOST_SP_DISABLE_THREADS)

# Override the boost index type to match the STL for Boost.MultiArray
# (std::ptrdiff_t to std::size_t)
# Note: This header guard hasn't changed since 2002 so it seems not too insane
# to rely on.
add_definitions(-DBOOST_MULTI_ARRAY_TYPES_RG071801_HPP)

# Also enable rigorous protocol conformance checks in debug builds. If we find
# that debug builds increase in compile time significantly we can disable this.
if (${SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE}
    OR "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  if (${SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE})
    message(STATUS "Enable rigorous protocol conformance checks")
  else()
    message(STATUS "Enable rigorous protocol conformance checks in Debug build")
  endif()
  add_definitions(-DSPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE)
endif()
