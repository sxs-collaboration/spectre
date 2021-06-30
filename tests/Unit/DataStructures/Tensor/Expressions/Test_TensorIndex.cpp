// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.TensorIndex",
                  "[DataStructures][Unit]") {
  // Test `make_tensorindex_list`
  CHECK(std::is_same_v<
        make_tensorindex_list<ti_j, ti_A, ti_b>,
        tmpl::list<std::decay_t<decltype(ti_j)>, std::decay_t<decltype(ti_A)>,
                   std::decay_t<decltype(ti_b)>>>);
  CHECK(std::is_same_v<make_tensorindex_list<>, tmpl::list<>>);

  // Test `get_tensorindex_value_with_opposite_valence`
  //
  // For (1) lower spacetime indices, (2) upper spacetime indices, (3) lower
  // spatial indices, and (4) upper spatial indices, the encoding of the index
  // with the same index type but opposite valence is checked for the following
  // cases: (i) the smallest encoding, (ii) the largest encoding, and (iii) a
  // value in between.

  // Lower spacetime
  CHECK(get_tensorindex_value_with_opposite_valence(0) ==
        TensorIndex_detail::upper_sentinel);
  CHECK(get_tensorindex_value_with_opposite_valence(
            TensorIndex_detail::upper_sentinel - 1) ==
        TensorIndex_detail::spatial_sentinel - 1);
  CHECK(get_tensorindex_value_with_opposite_valence(115) ==
        TensorIndex_detail::upper_sentinel + 115);

  // Upper spacetime
  CHECK(get_tensorindex_value_with_opposite_valence(
            TensorIndex_detail::upper_sentinel) == 0);
  CHECK(get_tensorindex_value_with_opposite_valence(
            TensorIndex_detail::spatial_sentinel - 1) ==
        TensorIndex_detail::upper_sentinel - 1);
  CHECK(get_tensorindex_value_with_opposite_valence(
            TensorIndex_detail::upper_sentinel + 88) == 88);

  // Lower spatial
  CHECK(get_tensorindex_value_with_opposite_valence(
            TensorIndex_detail::spatial_sentinel) ==
        TensorIndex_detail::upper_spatial_sentinel);
  CHECK(get_tensorindex_value_with_opposite_valence(
            TensorIndex_detail::upper_spatial_sentinel - 1) ==
        TensorIndex_detail::max_sentinel - 1);
  CHECK(get_tensorindex_value_with_opposite_valence(
            TensorIndex_detail::spatial_sentinel + 232) ==
        TensorIndex_detail::upper_spatial_sentinel + 232);

  // Upper spatial
  CHECK(get_tensorindex_value_with_opposite_valence(
            TensorIndex_detail::upper_spatial_sentinel) ==
        TensorIndex_detail::spatial_sentinel);
  CHECK(get_tensorindex_value_with_opposite_valence(
            TensorIndex_detail::max_sentinel - 1) ==
        TensorIndex_detail::upper_spatial_sentinel - 1);
  CHECK(get_tensorindex_value_with_opposite_valence(
            TensorIndex_detail::upper_spatial_sentinel + 3) ==
        TensorIndex_detail::spatial_sentinel + 3);
}
