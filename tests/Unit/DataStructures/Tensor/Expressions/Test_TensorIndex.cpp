// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.TensorIndex",
                  "[DataStructures][Unit]") {
  // Test `make_tensorindex_list`
  // Check at compile time since some other tests use this metafunction
  static_assert(std::is_same_v<make_tensorindex_list<ti::j, ti::A, ti::b>,
                               tmpl::list<std::decay_t<decltype(ti::j)>,
                                          std::decay_t<decltype(ti::A)>,
                                          std::decay_t<decltype(ti::b)>>>,
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
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(0) ==
        tenex::TensorIndex_detail::upper_sentinel);
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(
            tenex::TensorIndex_detail::upper_sentinel - 1) ==
        tenex::TensorIndex_detail::spatial_sentinel - 1);
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(115) ==
        tenex::TensorIndex_detail::upper_sentinel + 115);

  // Upper spacetime
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(
            tenex::TensorIndex_detail::upper_sentinel) == 0);
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(
            tenex::TensorIndex_detail::spatial_sentinel - 1) ==
        tenex::TensorIndex_detail::upper_sentinel - 1);
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(
            tenex::TensorIndex_detail::upper_sentinel + 88) == 88);

  // Lower spatial
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(
            tenex::TensorIndex_detail::spatial_sentinel) ==
        tenex::TensorIndex_detail::upper_spatial_sentinel);
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(
            tenex::TensorIndex_detail::upper_spatial_sentinel - 1) ==
        tenex::TensorIndex_detail::max_sentinel - 1);
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(
            tenex::TensorIndex_detail::spatial_sentinel + 232) ==
        tenex::TensorIndex_detail::upper_spatial_sentinel + 232);

  // Upper spatial
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(
            tenex::TensorIndex_detail::upper_spatial_sentinel) ==
        tenex::TensorIndex_detail::spatial_sentinel);
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(
            tenex::TensorIndex_detail::max_sentinel - 1) ==
        tenex::TensorIndex_detail::upper_spatial_sentinel - 1);
  CHECK(tenex::get_tensorindex_value_with_opposite_valence(
            tenex::TensorIndex_detail::upper_spatial_sentinel + 3) ==
        tenex::TensorIndex_detail::spatial_sentinel + 3);

  // Test tensorindex_list_is_valid
  CHECK(tenex::tensorindex_list_is_valid<make_tensorindex_list<>>::value);
  CHECK(tenex::tensorindex_list_is_valid<make_tensorindex_list<ti::J>>::value);
  CHECK(tenex::tensorindex_list_is_valid<
        make_tensorindex_list<ti::a, ti::c, ti::I, ti::B>>::value);
  CHECK(tenex::tensorindex_list_is_valid<
        make_tensorindex_list<ti::t, ti::T, ti::T, ti::T, ti::t>>::value);
  CHECK(tenex::tensorindex_list_is_valid<
        make_tensorindex_list<ti::d, ti::T, ti::D>>::value);
  CHECK(not tenex::tensorindex_list_is_valid<
        make_tensorindex_list<ti::I, ti::a, ti::I>>::value);

  // Test tensorindex_list_is_valid
  CHECK(tenex::tensorindex_list_is_valid<make_tensorindex_list<>>::value);
  CHECK(tenex::tensorindex_list_is_valid<make_tensorindex_list<ti::J>>::value);
  CHECK(tenex::tensorindex_list_is_valid<
        make_tensorindex_list<ti::a, ti::c, ti::I, ti::B>>::value);
  CHECK(tenex::tensorindex_list_is_valid<
        make_tensorindex_list<ti::t, ti::T, ti::T, ti::T, ti::t>>::value);
  CHECK(tenex::tensorindex_list_is_valid<
        make_tensorindex_list<ti::d, ti::T, ti::D>>::value);
  CHECK(not tenex::tensorindex_list_is_valid<
        make_tensorindex_list<ti::I, ti::a, ti::I>>::value);

  // Test generic_indices_at_same_positions
  CHECK(
      tenex::generic_indices_at_same_positions<make_tensorindex_list<>,
                                               make_tensorindex_list<>>::value);
  CHECK(tenex::generic_indices_at_same_positions<
        make_tensorindex_list<ti::a, ti::c, ti::I, ti::B>,
        make_tensorindex_list<ti::a, ti::c, ti::I, ti::B>>::value);
  CHECK(not tenex::generic_indices_at_same_positions<
        make_tensorindex_list<ti::a, ti::c, ti::I, ti::B>,
        make_tensorindex_list<ti::a, ti::c, ti::i, ti::B>>::value);
  CHECK(not tenex::generic_indices_at_same_positions<
        make_tensorindex_list<ti::a, ti::c, ti::I, ti::B>,
        make_tensorindex_list<ti::a, ti::c, ti::I>>::value);
  CHECK(not tenex::generic_indices_at_same_positions<
        make_tensorindex_list<ti::a, ti::c, ti::I>,
        make_tensorindex_list<ti::a, ti::c, ti::I, ti::B>>::value);
  CHECK(not tenex::generic_indices_at_same_positions<
        make_tensorindex_list<ti::j, ti::B>,
        make_tensorindex_list<ti::B, ti::j>>::value);
  CHECK(tenex::generic_indices_at_same_positions<
        make_tensorindex_list<ti::T>, make_tensorindex_list<ti::t>>::value);
  CHECK(tenex::generic_indices_at_same_positions<
        make_tensorindex_list<ti::t, ti::T, ti::T>,
        make_tensorindex_list<ti::T, ti::t, ti::T>>::value);
  CHECK(tenex::generic_indices_at_same_positions<
        make_tensorindex_list<ti::i, ti::t, ti::C>,
        make_tensorindex_list<ti::i, ti::T, ti::C>>::value);
}
