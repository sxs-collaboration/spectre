// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <type_traits>

#include "DataStructures/Tensor/Expressions/SpatialSpacetimeIndex.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"

namespace {
constexpr size_t dim = 3;
using frame = Frame::Grid;

using index_list_empty = index_list<>;
using index_list_a = index_list<SpacetimeIndex<dim, UpLo::Lo, frame>>;
using index_list_i = index_list<SpatialIndex<dim, UpLo::Lo, frame>>;
using index_list_abc = index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                  SpacetimeIndex<dim, UpLo::Lo, frame>,
                                  SpacetimeIndex<dim, UpLo::Lo, frame>>;
using index_list_aBC = index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                  SpacetimeIndex<dim, UpLo::Up, frame>,
                                  SpacetimeIndex<dim, UpLo::Up, frame>>;
using index_list_aBI = index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                  SpacetimeIndex<dim, UpLo::Up, frame>,
                                  SpatialIndex<dim, UpLo::Up, frame>>;
using index_list_aIB = index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                  SpatialIndex<dim, UpLo::Up, frame>,
                                  SpacetimeIndex<dim, UpLo::Up, frame>>;
using index_list_aIJ = index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                  SpatialIndex<dim, UpLo::Up, frame>,
                                  SpatialIndex<dim, UpLo::Up, frame>>;
using index_list_iAB = index_list<SpatialIndex<dim, UpLo::Lo, frame>,
                                  SpacetimeIndex<dim, UpLo::Up, frame>,
                                  SpacetimeIndex<dim, UpLo::Up, frame>>;
using index_list_iAJ = index_list<SpatialIndex<dim, UpLo::Lo, frame>,
                                  SpacetimeIndex<dim, UpLo::Up, frame>,
                                  SpatialIndex<dim, UpLo::Up, frame>>;
using index_list_iJA = index_list<SpatialIndex<dim, UpLo::Lo, frame>,
                                  SpatialIndex<dim, UpLo::Up, frame>,
                                  SpacetimeIndex<dim, UpLo::Up, frame>>;
using index_list_iJK = index_list<SpatialIndex<dim, UpLo::Lo, frame>,
                                  SpatialIndex<dim, UpLo::Up, frame>,
                                  SpatialIndex<dim, UpLo::Up, frame>>;

using ti_list_empty = tmpl::list<>;
using ti_list_a = make_tensorindex_list<ti_a>;
using ti_list_i = make_tensorindex_list<ti_i>;
using ti_list_abc = make_tensorindex_list<ti_a, ti_b, ti_c>;
using ti_list_ijk = make_tensorindex_list<ti_i, ti_j, ti_k>;
using ti_list_abi = make_tensorindex_list<ti_a, ti_b, ti_i>;
using ti_list_aij = make_tensorindex_list<ti_a, ti_i, ti_j>;
using ti_list_iaj = make_tensorindex_list<ti_i, ti_a, ti_j>;
using ti_list_ija = make_tensorindex_list<ti_i, ti_j, ti_a>;

using positions_list_empty = tmpl::integral_list<size_t>;
using positions_list_0 = tmpl::integral_list<size_t, 0>;
using positions_list_1 = tmpl::integral_list<size_t, 1>;
using positions_list_2 = tmpl::integral_list<size_t, 2>;
using positions_list_01 = tmpl::integral_list<size_t, 0, 1>;
using positions_list_12 = tmpl::integral_list<size_t, 1, 2>;
using positions_list_02 = tmpl::integral_list<size_t, 0, 2>;
using positions_list_012 = tmpl::integral_list<size_t, 0, 1, 2>;

void test_spatial_spacetime_index_positions() {
  CHECK(std::is_same_v<
        TensorExpressions::detail::spatial_spacetime_index_positions<
            index_list_empty, ti_list_empty>,
        positions_list_empty>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::spatial_spacetime_index_positions<
            index_list_a, ti_list_a>,
        positions_list_empty>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::spatial_spacetime_index_positions<
            index_list_a, ti_list_i>,
        positions_list_0>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::spatial_spacetime_index_positions<
            index_list_aBC, ti_list_abc>,
        positions_list_empty>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::spatial_spacetime_index_positions<
            index_list_aBC, ti_list_iaj>,
        positions_list_02>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::spatial_spacetime_index_positions<
            index_list_aBC, ti_list_abc>,
        positions_list_empty>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::spatial_spacetime_index_positions<
            index_list_aBC, ti_list_abi>,
        positions_list_2>);
}

void test_get_spatial_spacetime_index_symmetry() {
  constexpr std::array<std::int32_t, 0> symm_empty = {{}};
  constexpr std::array<std::int32_t, 1> symm_1 = {{1}};
  constexpr std::array<std::int32_t, 3> symm_111 = {{1, 1, 1}};
  constexpr std::array<std::int32_t, 3> symm_211 = {{2, 1, 1}};
  constexpr std::array<std::int32_t, 3> symm_121 = {{1, 2, 1}};
  constexpr std::array<std::int32_t, 3> symm_221 = {{2, 2, 1}};
  constexpr std::array<std::int32_t, 3> symm_321 = {{3, 2, 1}};

  constexpr std::array<size_t, 0> positions_empty = {{}};
  constexpr std::array<size_t, 1> positions_0 = {{0}};
  constexpr std::array<size_t, 1> positions_1 = {{1}};
  constexpr std::array<size_t, 1> positions_2 = {{2}};
  constexpr std::array<size_t, 2> positions_01 = {{0, 1}};
  constexpr std::array<size_t, 2> positions_02 = {{0, 2}};
  constexpr std::array<size_t, 2> positions_12 = {{1, 2}};
  constexpr std::array<size_t, 3> positions_012 = {{0, 1, 2}};

  // Rank 0
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_empty, positions_empty)) == detail::symmetry(symm_empty));

  // Rank 1
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_1, positions_empty)) == detail::symmetry(symm_1));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_1, positions_0)) == detail::symmetry(symm_1));

  // Rank 3, input symmetry with three symmetric indices
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_111, positions_0)) == detail::symmetry(symm_211));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_111, positions_1)) == detail::symmetry(symm_121));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_111, positions_2)) == detail::symmetry(symm_221));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_111, positions_01)) == detail::symmetry(symm_221));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_111, positions_02)) == detail::symmetry(symm_121));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_111, positions_12)) == detail::symmetry(symm_211));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_111, positions_012)) == detail::symmetry(symm_111));

  // Rank 3, input symmetry with two symmetric indices
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_121, positions_empty)) == detail::symmetry(symm_121));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_121, positions_0)) == detail::symmetry(symm_321));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_121, positions_1)) == detail::symmetry(symm_121));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_121, positions_2)) == detail::symmetry(symm_321));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_121, positions_01)) == detail::symmetry(symm_321));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_121, positions_02)) == detail::symmetry(symm_121));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_121, positions_12)) == detail::symmetry(symm_321));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_121, positions_012)) == detail::symmetry(symm_121));

  // Rank 3, input symmetry with no symmetric indices
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_321, positions_empty)) == detail::symmetry(symm_321));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_321, positions_0)) == detail::symmetry(symm_321));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_321, positions_1)) == detail::symmetry(symm_321));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_321, positions_2)) == detail::symmetry(symm_321));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_321, positions_01)) == detail::symmetry(symm_321));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_321, positions_02)) == detail::symmetry(symm_321));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_321, positions_12)) == detail::symmetry(symm_321));
  CHECK(detail::symmetry(
            TensorExpressions::detail::get_spatial_spacetime_index_symmetry(
                symm_321, positions_012)) == detail::symmetry(symm_321));
}

void test_replace_spatial_spacetime_indices() {
  // Rank 0
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_empty, positions_list_empty>,
        index_list_empty>);

  // Rank 1
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_a, positions_list_empty>,
        index_list_a>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_a, positions_list_0>,
        index_list_i>);

  // Rank 3, one spacetime index
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_iJA, positions_list_empty>,
        index_list_iJA>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_iJA, positions_list_2>,
        index_list_iJK>);

  // Rank 3, two spacetime indices
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_iAB, positions_list_empty>,
        index_list_iAB>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_iAB, positions_list_1>,
        index_list_iJA>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_iAB, positions_list_2>,
        index_list_iAJ>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_iAB, positions_list_12>,
        index_list_iJK>);

  // Rank 3, three spacetime indices
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_aBC, positions_list_empty>,
        index_list_aBC>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_aBC, positions_list_0>,
        index_list_iAB>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_aBC, positions_list_1>,
        index_list_aIB>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_aBC, positions_list_2>,
        index_list_aBI>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_aBC, positions_list_01>,
        index_list_iJA>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_aBC, positions_list_02>,
        index_list_iAJ>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_aBC, positions_list_12>,
        index_list_aIJ>);
  CHECK(std::is_same_v<
        TensorExpressions::detail::replace_spatial_spacetime_indices<
            index_list_aBC, positions_list_012>,
        index_list_iJK>);
}

void test_spatial_spacetime_index_transformation_from_positions() {
  constexpr std::array<size_t, 0> positions_empty = {{}};
  constexpr std::array<size_t, 1> positions_0 = {{0}};
  constexpr std::array<size_t, 1> positions_1 = {{1}};
  constexpr std::array<size_t, 1> positions_2 = {{2}};
  constexpr std::array<size_t, 2> positions_12 = {{1, 2}};

  constexpr std::array<std::int32_t, 0> transformation_empty = {{}};
  constexpr std::array<std::int32_t, 1> transformation_0 = {{0}};
  constexpr std::array<std::int32_t, 1> transformation_m1 = {{-1}};
  constexpr std::array<std::int32_t, 1> transformation_p1 = {{1}};
  constexpr std::array<std::int32_t, 3> transformation_000 = {{0, 0, 0}};
  constexpr std::array<std::int32_t, 3> transformation_m10p10 = {{-1, 0, 1}};
  constexpr std::array<std::int32_t, 3> transformation_00m1 = {{0, 0, -1}};

  // Rank 0
  CHECK(TensorExpressions::detail::
            spatial_spacetime_index_transformation_from_positions<0>(
                positions_empty, positions_empty) == transformation_empty);

  // Rank 1
  CHECK(TensorExpressions::detail::
            spatial_spacetime_index_transformation_from_positions<1>(
                positions_empty, positions_empty) == transformation_0);
  CHECK(TensorExpressions::detail::
            spatial_spacetime_index_transformation_from_positions<1>(
                positions_0, positions_0) == transformation_0);
  CHECK(TensorExpressions::detail::
            spatial_spacetime_index_transformation_from_positions<1>(
                positions_0, positions_empty) == transformation_m1);
  CHECK(TensorExpressions::detail::
            spatial_spacetime_index_transformation_from_positions<1>(
                positions_empty, positions_0) == transformation_p1);

  // Rank 3
  CHECK(TensorExpressions::detail::
            spatial_spacetime_index_transformation_from_positions<3>(
                positions_empty, positions_empty) == transformation_000);
  CHECK(TensorExpressions::detail::
            spatial_spacetime_index_transformation_from_positions<3>(
                positions_0, positions_2) == transformation_m10p10);
  CHECK(TensorExpressions::detail::
            spatial_spacetime_index_transformation_from_positions<3>(
                positions_12, positions_1) == transformation_00m1);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.SpatialSpacetimeIndex",
                  "[DataStructures][Unit]") {
  test_spatial_spacetime_index_positions();
  test_get_spatial_spacetime_index_symmetry();
  test_replace_spatial_spacetime_indices();
  test_spatial_spacetime_index_transformation_from_positions();
}
