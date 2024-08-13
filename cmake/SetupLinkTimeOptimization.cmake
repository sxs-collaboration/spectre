# Distributed under the MIT License.
# See LICENSE.txt for details.

option(SPECTRE_LTO "Use link-time optimization if available" OFF)

include(CheckIPOSupported)
check_ipo_supported(RESULT _RESULT OUTPUT _OUTPUT)
if(_RESULT AND SPECTRE_LTO)
  option(SPECTRE_LTO_CORES "Number of cores to use for LTO" OFF)
  if (NOT SPECTRE_LTO_CORES)
    set(SPECTRE_LTO_CORES "auto")
  endif()
  target_link_options(SpectreFlags
    INTERFACE
    -flto=${SPECTRE_LTO_CORES})
  message(STATUS "Link-time optimizations enabled with ${SPECTRE_LTO_CORES}")
else()
  target_link_options(SpectreFlags
    INTERFACE
    -fno-lto)
  message(STATUS "Link-time optimizations disabled (no LTO)")
endif()
