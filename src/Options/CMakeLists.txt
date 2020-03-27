# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Options)

add_spectre_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  ErrorHandling
  YamlCpp
  Utilities
  )
