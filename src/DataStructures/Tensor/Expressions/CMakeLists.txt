# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AddSubtract.hpp
  Contract.hpp
  Divide.hpp
  Evaluate.hpp
  IndexPropertyCheck.hpp
  LhsTensorSymmAndIndices.hpp
  Negate.hpp
  NumberAsExpression.hpp
  Product.hpp
  SpatialSpacetimeIndex.hpp
  SquareRoot.hpp
  TensorAsExpression.hpp
  TensorExpression.hpp
  TensorIndex.hpp
  TensorIndexTransformation.hpp
  TimeIndex.hpp
  )
