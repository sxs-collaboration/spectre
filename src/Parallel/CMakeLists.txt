# Distributed under the MIT License.
# See LICENSE.txt for details.

add_charm_module(ConstGlobalCache)
add_charm_module(Main)

set(LIBRARY Parallel)

add_spectre_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  DataStructures
  ErrorHandling
  Informer
  Options
  Utilities
  )
