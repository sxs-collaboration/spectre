// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.TensorIndex",
                  "[DataStructures][Unit]") {
  // Test `make_tensorindex_list`
  // Check at compile time since some other tests use this metafunction
  static_assert(
      std::is_same_v<
          make_tensorindex_list<ti_j, ti_A, ti_b>,
          tmpl::list<std::decay_t<decltype(ti_j)>, std::decay_t<decltype(ti_A)>,
                     std::decay_t<decltype(ti_b)>>>,
      "make_tensorindex_list failed for non-empty list");
  static_assert(std::is_same_v<make_tensorindex_list<>, tmpl::list<>>,
                "make_tensorindex_list failed for empty list");

  // Test `get_tensorindex_value_with_opposite_valence`
  //
  // For (1) lower spacetime indices, (2) upper spacetime indices, (3) lower
  // spatial indices, and (4) upper spatial indices, the encoding of the index
  // with the same index type but opposite valence is checked for the following
  // cases: (i) the smallest encoding, (ii) the largest encoding, and (iii) a
  // value in between.

  // Lower spacetime
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(0) ==
        TensorExpressions::TensorIndex_detail::upper_sentinel);
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(
            TensorExpressions::TensorIndex_detail::upper_sentinel - 1) ==
        TensorExpressions::TensorIndex_detail::spatial_sentinel - 1);
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(115) ==
        TensorExpressions::TensorIndex_detail::upper_sentinel + 115);

  // Upper spacetime
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(
            TensorExpressions::TensorIndex_detail::upper_sentinel) == 0);
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(
            TensorExpressions::TensorIndex_detail::spatial_sentinel - 1) ==
        TensorExpressions::TensorIndex_detail::upper_sentinel - 1);
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(
            TensorExpressions::TensorIndex_detail::upper_sentinel + 88) == 88);

  // Lower spatial
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(
            TensorExpressions::TensorIndex_detail::spatial_sentinel) ==
        TensorExpressions::TensorIndex_detail::upper_spatial_sentinel);
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(
            TensorExpressions::TensorIndex_detail::upper_spatial_sentinel -
            1) == TensorExpressions::TensorIndex_detail::max_sentinel - 1);
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(
            TensorExpressions::TensorIndex_detail::spatial_sentinel + 232) ==
        TensorExpressions::TensorIndex_detail::upper_spatial_sentinel + 232);

  // Upper spatial
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(
            TensorExpressions::TensorIndex_detail::upper_spatial_sentinel) ==
        TensorExpressions::TensorIndex_detail::spatial_sentinel);
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(
            TensorExpressions::TensorIndex_detail::max_sentinel - 1) ==
        TensorExpressions::TensorIndex_detail::upper_spatial_sentinel - 1);
  CHECK(TensorExpressions::get_tensorindex_value_with_opposite_valence(
            TensorExpressions::TensorIndex_detail::upper_spatial_sentinel +
            3) == TensorExpressions::TensorIndex_detail::spatial_sentinel + 3);
}
