# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AddSubtract.hpp
  Contract.hpp
  Evaluate.hpp
  LhsTensorSymmAndIndices.hpp
  Product.hpp
  NumberAsExpression.hpp
  SpatialSpacetimeIndex.hpp
  SquareRoot.hpp
  TensorAsExpression.hpp
  TensorExpression.hpp
  TensorIndex.hpp
  TensorIndexTransformation.hpp
  TimeIndex.hpp
  )
