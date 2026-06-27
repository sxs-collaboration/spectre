# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY TensorYlm)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CartToSphere.cpp
  Filter.cpp
  Helpers.cpp
  SphereToCart.cpp
  WignerThreeJ.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApplyFilter.hpp
  ApplyFilter.tpp
  CartToSphere.hpp
  Filter.hpp
  Helpers.hpp
  SphereToCart.hpp
  TensorYlm.hpp
  WignerThreeJ.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  SphericalHarmonics
  Utilities
  )
