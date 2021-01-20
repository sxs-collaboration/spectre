# Distributed under the MIT License.
# See LICENSE.txt for details.

option(USE_CCACHE "Use CCache if available of speed up builds" ON)
if(USE_CCACHE)
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    string(CONCAT CCACHE_COMMAND
      "CCACHE_SLOPPINESS=pch_defines,time_macros "
      # In order to make ccache work with precompiled headers we need to:
      # - Hash the header file in the repo that will generate the precompiled
      #   header
      # - In the past we had to ignore the precompiled header in the build
      #   directory:
      #   "CCACHE_IGNOREHEADERS=${CMAKE_BINARY_DIR}/SpectrePch.hpp:${CMAKE_BINARY_DIR}/SpectrePch.hpp.gch "
      #   This doesn't seem to be necessary anymore, but may help if issues
      #   related to PCH and ccache come up in the future.
      "CCACHE_EXTRAFILES=${CMAKE_SOURCE_DIR}/tools/SpectrePch.hpp "
      "ccache"
    )
    # CCache offers no benefit for linking
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_COMMAND})
    message(STATUS "Using ccache for compilation. It is invoked as: ${CCACHE_COMMAND}")
  else()
    message(STATUS "Could not find ccache")
  endif(CCACHE_FOUND)
endif(USE_CCACHE)
