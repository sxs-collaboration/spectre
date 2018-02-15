# Distributed under the MIT License.
# See LICENSE.txt for details.

option(USE_CCACHE "Use CCache if available of speed up builds" ON)
if(USE_CCACHE)
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    # CCache offers no benefit for linking
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE
      "CCACHE_SLOPPINESS=pch_defines,time_macros ccache")
    message(STATUS "Using ccache for compilation")
  else()
    message(STATUS "Could not find ccache")
  endif(CCACHE_FOUND)
endif(USE_CCACHE)
