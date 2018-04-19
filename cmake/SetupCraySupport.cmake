# Distributed under the MIT License.
# See LICENSE.txt for details.

# The cray compiler wrappers don't support -isystem and so on cray machines
# everything is included using -I. However, -I takes precedence over -isystem,
# so to make sure we can properly set include paths to override system include
# paths we must use -I.
option(USE_SYSTEM_INCLUDE "Use -isystem instead of -I" ON)
function(spectre_include_directories)
  if(${USE_SYSTEM_INCLUDE})
    include_directories(SYSTEM ${ARGN})
  else()
    include_directories(${ARGN})
  endif(${USE_SYSTEM_INCLUDE})
endfunction()
