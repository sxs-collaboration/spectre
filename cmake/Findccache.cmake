# Distributed under the MIT License.
# See LICENSE.txt for details.

find_program(CCACHE_EXEC ccache)

if (CCACHE_EXEC)
  # Get version
  execute_process(COMMAND ${CCACHE_EXEC} --version
    OUTPUT_VARIABLE CCACHE_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  # Keep only first line of output
  string(REGEX REPLACE "\n.*" "" CCACHE_VERSION ${CCACHE_VERSION})
  # Remove "ccache version " prefix
  string(REGEX REPLACE "ccache version " "" CCACHE_VERSION ${CCACHE_VERSION})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  ccache REQUIRED_VARS CCACHE_EXEC VERSION_VAR CCACHE_VERSION
  )
mark_as_advanced(CCACHE_VERSION)
