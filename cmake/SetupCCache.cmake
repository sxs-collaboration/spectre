# Distributed under the MIT License.
# See LICENSE.txt for details.

option(USE_CCACHE "Use CCache if available of speed up builds" OFF)
if(USE_CCACHE)
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    message(STATUS "Using ccache for compilation")
  else()
    message(STATUS "Could not find ccache")
  endif(CCACHE_FOUND)
endif(USE_CCACHE)
