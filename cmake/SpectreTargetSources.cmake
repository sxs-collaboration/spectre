# Distributed under the MIT License.
# See LICENSE.txt for details.

# Alias for target_sources.
function(spectre_target_sources TARGET)
  target_sources(${TARGET} ${ARGN})
endfunction()
