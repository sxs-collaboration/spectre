# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Expressions")

set(LIBRARY_SOURCES
  Test_AddSubSymmetry.cpp
  Test_AddSubtract.cpp
  Test_Contract.cpp
  Test_Divide.cpp
  Test_Evaluate.cpp
  Test_EvaluateRank3NonSymmetric.cpp
  Test_EvaluateRank3Symmetric.cpp
  Test_EvaluateRank4.cpp
  Test_EvaluateSpatialSpacetimeIndex.cpp
  Test_EvaluateTimeIndex.cpp
  Test_MixedOperations.cpp
  Test_Negate.cpp
  Test_Product.cpp
  Test_ProductHighRankIntermediate.cpp
  Test_SpatialSpacetimeIndex.cpp
  Test_SquareRoot.cpp
  Test_TensorIndex.cpp
  Test_TensorIndexTransformation.cpp
  Test_TimeIndex.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
