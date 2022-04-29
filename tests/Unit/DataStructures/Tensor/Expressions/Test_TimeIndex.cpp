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
  CHECK(tt::is_time_index<std::decay_t<decltype(ti::t)>>::value);
  CHECK(tt::is_time_index<std::decay_t<decltype(ti::T)>>::value);
  CHECK(not tt::is_time_index<std::decay_t<decltype(ti::b)>>::value);
  CHECK(not tt::is_time_index<std::decay_t<decltype(ti::I)>>::value);

  // Test is_time_index_value
  CHECK(tenex::detail::is_time_index_value(ti::t.value));
  CHECK(tenex::detail::is_time_index_value(ti::T.value));
  CHECK(not tenex::detail::is_time_index_value(ti::C.value));
  CHECK(not tenex::detail::is_time_index_value(ti::j.value));

  // Lists of TensorIndexs used for testing below
  using empty_index_list = make_tensorindex_list<>;
  using a = make_tensorindex_list<ti::a>;
  using aiC = make_tensorindex_list<ti::a, ti::i, ti::C>;
  using t = make_tensorindex_list<ti::t>;
  using TT = make_tensorindex_list<ti::T, ti::T>;
  using tc = make_tensorindex_list<ti::t, ti::c>;
  using c = make_tensorindex_list<ti::c>;
  using atTBjT =
      make_tensorindex_list<ti::a, ti::t, ti::T, ti::B, ti::j, ti::T>;
  using aBj = make_tensorindex_list<ti::a, ti::B, ti::j>;

  // Test remove_time_indices
  CHECK(
      std::is_same_v<tenex::detail::remove_time_indices<empty_index_list>::type,
                     empty_index_list>);
  CHECK(std::is_same_v<tenex::detail::remove_time_indices<a>::type, a>);
  CHECK(std::is_same_v<tenex::detail::remove_time_indices<aiC>::type, aiC>);
  CHECK(std::is_same_v<tenex::detail::remove_time_indices<t>::type,
                       empty_index_list>);
  CHECK(std::is_same_v<tenex::detail::remove_time_indices<TT>::type,
                       empty_index_list>);
  CHECK(std::is_same_v<tenex::detail::remove_time_indices<tc>::type, c>);
  CHECK(std::is_same_v<tenex::detail::remove_time_indices<atTBjT>::type, aBj>);

  // Test get_time_index_positions
  const std::array<size_t, 0> expected_empty_positions{};
  CHECK(tenex::detail::get_time_index_positions<empty_index_list>() ==
        expected_empty_positions);
  CHECK(tenex::detail::get_time_index_positions<a>() ==
        expected_empty_positions);
  CHECK(tenex::detail::get_time_index_positions<aiC>() ==
        expected_empty_positions);
  const std::array<size_t, 1> expected_t_positions{0};
  CHECK(tenex::detail::get_time_index_positions<t>() == expected_t_positions);
  const std::array<size_t, 2> expected_TT_positions{0, 1};
  CHECK(tenex::detail::get_time_index_positions<TT>() == expected_TT_positions);
  const std::array<size_t, 1> expected_tc_positions{0};
  CHECK(tenex::detail::get_time_index_positions<tc>() == expected_tc_positions);
  const std::array<size_t, 3> expected_atTBjT_positions{1, 2, 5};
  CHECK(tenex::detail::get_time_index_positions<atTBjT>() ==
        expected_atTBjT_positions);
}
