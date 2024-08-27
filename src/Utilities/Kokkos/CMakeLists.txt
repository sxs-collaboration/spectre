# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SpectreKokkos)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  KokkosCore.hpp
)

if(TARGET Kokkos::kokkos)
  target_link_libraries(
    ${LIBRARY}
    INTERFACE
    Kokkos::kokkos
  )
endif()
