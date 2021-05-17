# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GhGrMhdSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  InstantiateWrappedGr.cpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  GeneralRelativitySolutions
  GrMhdSolutions
  )
