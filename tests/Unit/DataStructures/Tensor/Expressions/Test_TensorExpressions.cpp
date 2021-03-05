// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.GetOppositeValence",
                  "[DataStructures][Unit]") {
  // For (1) lower spacetime indices, (2) upper spacetime indices, (3) lower
  // spatial indices, and (4) upper spatial indices, the encoding of the index
  // with the same index type but opposite valence is checked for the following
  // cases: (i) the smallest encoding, (ii) the largest encoding, and (iii) a
  // value in between.

  // Lower spacetime
  CHECK(get_tensorindex_value_with_opposite_valence(0) == upper_sentinel);
  CHECK(get_tensorindex_value_with_opposite_valence(upper_sentinel - 1) ==
        spatial_sentinel - 1);
  CHECK(get_tensorindex_value_with_opposite_valence(115) ==
        upper_sentinel + 115);

  // Upper spacetime
  CHECK(get_tensorindex_value_with_opposite_valence(upper_sentinel) == 0);
  CHECK(get_tensorindex_value_with_opposite_valence(spatial_sentinel - 1) ==
        upper_sentinel - 1);
  CHECK(get_tensorindex_value_with_opposite_valence(upper_sentinel + 88) == 88);

  // Lower spatial
  CHECK(get_tensorindex_value_with_opposite_valence(spatial_sentinel) ==
        upper_spatial_sentinel);
  CHECK(get_tensorindex_value_with_opposite_valence(upper_spatial_sentinel -
                                                    1) == max_sentinel - 1);
  CHECK(get_tensorindex_value_with_opposite_valence(spatial_sentinel + 232) ==
        upper_spatial_sentinel + 232);

  // Upper spatial
  CHECK(get_tensorindex_value_with_opposite_valence(upper_spatial_sentinel) ==
        spatial_sentinel);
  CHECK(get_tensorindex_value_with_opposite_valence(max_sentinel - 1) ==
        upper_spatial_sentinel - 1);
  CHECK(get_tensorindex_value_with_opposite_valence(upper_spatial_sentinel +
                                                    3) == spatial_sentinel + 3);
}
