# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EllipticSystems)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  GetSourcesComputer.hpp
  )

add_subdirectory(Elasticity)
add_subdirectory(Poisson)
add_subdirectory(Xcts)
