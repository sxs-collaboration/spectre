# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ErrorHandling)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AbortWithErrorMessage.cpp
  Breakpoint.cpp
  FloatingPointExceptions.cpp
  FormatStacktrace.cpp
  SegfaultHandler.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AbortWithErrorMessage.hpp
  Assert.hpp
  Breakpoint.hpp
  Error.hpp
  Exceptions.hpp
  ExpectsAndEnsures.hpp
  FloatingPointExceptions.hpp
  FormatStacktrace.hpp
  SegfaultHandler.hpp
  StaticAssert.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  Charmxx::charmxx
  PUBLIC
  SystemUtilities
  )

# Try to use libbacktrace for boost::stacktrace so it displays line numbers.
# The build configuration is explained here:
# https://www.boost.org/doc/libs/1_78_0/doc/html/stacktrace/configuration_and_build.html
# Note that we don't try to use Boost's addr2line mode because it takes a long
# time to run and doesn't work well:
# https://github.com/boostorg/stacktrace/issues/97
# On some systems (Debian) libbacktrace comes preinstalled with GCC (see notes
# in the Boost docs linked above).
find_library(
  BACKTRACE_LIB
  NAMES backtrace
  PATH_SUFFIXES lib64 lib
  HINTS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
find_path(
  BACKTRACE_HEADER_DIR
  NAMES backtrace.h
  PATH_SUFFIXES include
  HINTS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
if (NOT BACKTRACE_LIB STREQUAL "BACKTRACE_LIB-NOTFOUND"
    AND NOT BACKTRACE_HEADER_DIR STREQUAL "BACKTRACE_HEADER_DIR-NOTFOUND")
  message(STATUS "Using libbacktrace for stack traces: ${BACKTRACE_LIB}")
  set(BACKTRACE_HEADER "<${BACKTRACE_HEADER_DIR}/backtrace.h>")
  target_compile_definitions(${LIBRARY} PRIVATE
    BOOST_STACKTRACE_USE_BACKTRACE
    BOOST_STACKTRACE_BACKTRACE_INCLUDE_FILE=${BACKTRACE_HEADER})
  target_link_libraries(${LIBRARY} PRIVATE ${BACKTRACE_LIB})
else()
  # Fall back to "basic" mode. It needs no configuration because it's Boost's
  # default.
  message(STATUS "Using basic mode for stack traces. Install libbacktrace "
    "(https://github.com/ianlancetaylor/libbacktrace) to obtain more useful "
    "traces in errors and asserts.")
endif()
# All modes use libdl (see https://www.boost.org/doc/libs/1_78_0/doc/html/stacktrace/configuration_and_build.html)
# CMake provides the correct way to link with libdl.
target_link_libraries(${LIBRARY} PRIVATE ${CMAKE_DL_LIBS})

# Boost::stacktrace needs the `_Unwind_Backtrace` function to be available.
# On macOS it is available without defining `_GNU_SOURCE`, so Boost wants us
# to define `BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED`. Otherwise we get a
# compiler error telling us to define this.
if (APPLE)
  target_compile_definitions(${LIBRARY} PRIVATE
    BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)
endif()
