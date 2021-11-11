# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  TensorData.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ExtractPoint.hpp
  Identity.hpp
  IndexType.hpp
  Metafunctions.hpp
  Slice.hpp
  Structure.hpp
  Symmetry.hpp
  Tensor.hpp
  TensorData.hpp
  TypeAliases.hpp
  )

add_subdirectory(EagerMath)
add_subdirectory(Expressions)
