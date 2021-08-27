// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.TimeIndex",
                  "[DataStructures][Unit]") {
  // Test tt::is_time_index
  CHECK(tt::is_time_index<std::decay_t<decltype(ti_t)>>::value);
  CHECK(tt::is_time_index<std::decay_t<decltype(ti_T)>>::value);
  CHECK(not tt::is_time_index<std::decay_t<decltype(ti_b)>>::value);
  CHECK(not tt::is_time_index<std::decay_t<decltype(ti_I)>>::value);

  // Test is_time_index_value
  CHECK(TensorExpressions::detail::is_time_index_value(ti_t.value));
  CHECK(TensorExpressions::detail::is_time_index_value(ti_T.value));
  CHECK(not TensorExpressions::detail::is_time_index_value(ti_C.value));
  CHECK(not TensorExpressions::detail::is_time_index_value(ti_j.value));

  // Lists of TensorIndexs used for testing below
  using empty_index_list = make_tensorindex_list<>;
  using a = make_tensorindex_list<ti_a>;
  using aiC = make_tensorindex_list<ti_a, ti_i, ti_C>;
  using t = make_tensorindex_list<ti_t>;
  using TT = make_tensorindex_list<ti_T, ti_T>;
  using tc = make_tensorindex_list<ti_t, ti_c>;
  using c = make_tensorindex_list<ti_c>;
  using atTBjT = make_tensorindex_list<ti_a, ti_t, ti_T, ti_B, ti_j, ti_T>;
  using aBj = make_tensorindex_list<ti_a, ti_B, ti_j>;

  // Test remove_time_indices
  CHECK(std::is_same_v<
        TensorExpressions::detail::remove_time_indices<empty_index_list>::type,
        empty_index_list>);
  CHECK(std::is_same_v<TensorExpressions::detail::remove_time_indices<a>::type,
                       a>);
  CHECK(
      std::is_same_v<TensorExpressions::detail::remove_time_indices<aiC>::type,
                     aiC>);
  CHECK(std::is_same_v<TensorExpressions::detail::remove_time_indices<t>::type,
                       empty_index_list>);
  CHECK(std::is_same_v<TensorExpressions::detail::remove_time_indices<TT>::type,
                       empty_index_list>);
  CHECK(std::is_same_v<TensorExpressions::detail::remove_time_indices<tc>::type,
                       c>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::remove_time_indices<atTBjT>::type, aBj>);

  // Test get_time_index_positions
  const std::array<size_t, 0> expected_empty_positions{};
  CHECK(
      TensorExpressions::detail::get_time_index_positions<empty_index_list>() ==
      expected_empty_positions);
  CHECK(TensorExpressions::detail::get_time_index_positions<a>() ==
        expected_empty_positions);
  CHECK(TensorExpressions::detail::get_time_index_positions<aiC>() ==
        expected_empty_positions);
  const std::array<size_t, 1> expected_t_positions{0};
  CHECK(TensorExpressions::detail::get_time_index_positions<t>() ==
        expected_t_positions);
  const std::array<size_t, 2> expected_TT_positions{0, 1};
  CHECK(TensorExpressions::detail::get_time_index_positions<TT>() ==
        expected_TT_positions);
  const std::array<size_t, 1> expected_tc_positions{0};
  CHECK(TensorExpressions::detail::get_time_index_positions<tc>() ==
        expected_tc_positions);
  const std::array<size_t, 3> expected_atTBjT_positions{1, 2, 5};
  CHECK(TensorExpressions::detail::get_time_index_positions<atTBjT>() ==
        expected_atTBjT_positions);
}
