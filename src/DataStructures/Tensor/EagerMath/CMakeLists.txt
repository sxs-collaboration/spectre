# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  DataStructures
  PRIVATE
  OrthonormalOneform.cpp
  )

spectre_target_headers(
  DataStructures
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CrossProduct.hpp
  Determinant.hpp
  DeterminantAndInverse.hpp
  DotProduct.hpp
  Magnitude.hpp
  Norms.hpp
  OrthonormalOneform.hpp
  )
