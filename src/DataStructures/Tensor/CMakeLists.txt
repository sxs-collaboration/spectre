# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  DataStructures
  PRIVATE
  TensorData.cpp
  )

spectre_target_headers(
  DataStructures
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Identity.hpp
  IndexType.hpp
  MathFunctions.hpp
  Slice.hpp
  Structure.hpp
  Symmetry.hpp
  Tensor.hpp
  TensorData.hpp
  TypeAliases.hpp
  )


add_subdirectory(EagerMath)
add_subdirectory(Expressions)
