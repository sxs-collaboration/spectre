# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  DataStructures
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AddSubtract.hpp
  Contract.hpp
  Evaluate.hpp
  LhsTensorSymmAndIndices.hpp
  Product.hpp
  TensorAsExpression.hpp
  TensorExpression.hpp
  )
