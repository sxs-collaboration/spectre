// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank0.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank1.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <std::int32_t... Is>
using expected_symm = tmpl::integral_list<std::int32_t, Is...>;

template <IndexType... Is>
using indextype_list = tmpl::integral_list<IndexType, Is...>;

const IndexType spatial_index = IndexType::Spatial;
const IndexType spacetime_index = IndexType::Spacetime;

template <auto&... TensorIndices>
void test_contains_indices_to_contract_impl(const bool expected) {
  CHECK(tenex::detail::contains_indices_to_contract<sizeof...(TensorIndices)>(
            {{std::decay_t<decltype(TensorIndices)>::value...}}) == expected);
}

// Tests the helper function `tenex::detail::contains_indices_to_contract`
// correctly determines whether or not a list of tensor indices contains at
// least one index pair to contract
void test_contains_indices_to_contract() {
  test_contains_indices_to_contract_impl<ti::a, ti::b, ti::c>(false);
  test_contains_indices_to_contract_impl<ti::I, ti::j>(false);
  test_contains_indices_to_contract_impl<ti::j>(false);
  test_contains_indices_to_contract_impl(false);

  test_contains_indices_to_contract_impl<ti::d, ti::D>(true);
  test_contains_indices_to_contract_impl<ti::I, ti::i>(true);
  test_contains_indices_to_contract_impl<ti::a, ti::K, ti::B, ti::b>(true);
  test_contains_indices_to_contract_impl<ti::j, ti::c, ti::J, ti::A, ti::a>(
      true);
}

// Tests that the canonical ordering of symmetry values by `Symmetry` is
// consistent with what `tenex::detail::get_reordered_tensorindex_values`
// expects, which is that the symmetry values assigned to indepdenent indices
// is ascending from the rightmost position moving leftward with the rightmost
// symmetry value starting at 1
void test_lhs_tensorindex_reorder_symm_consistency() {
  const std::string error_msg =
      "tenex::detail::get_reordered_tensorindex_values() assumes a canonical "
      "form for Symmetry that is no longer the actual canonical form of "
      "Symmetry. The logic of this unit test and "
      "tenex::detail::get_reordered_tensorindex_values() must be updated to "
      "agree with the current canonical form for Symmetry";
  INFO(error_msg);

  CHECK(std::is_same_v<Symmetry<>, expected_symm<>>);
  CHECK(std::is_same_v<Symmetry<4>, expected_symm<1>>);
  CHECK(std::is_same_v<Symmetry<1, 2>, expected_symm<2, 1>>);
  CHECK(std::is_same_v<Symmetry<3, 5>, expected_symm<2, 1>>);
  CHECK(std::is_same_v<Symmetry<2, 2, 2>, expected_symm<1, 1, 1>>);
  CHECK(std::is_same_v<Symmetry<8, 4, 5, 5, 8>, expected_symm<1, 3, 2, 2, 1>>);
}

// Tests that the canonical ordering of multi-indices by
// `Tensor_detail::Structure` is consistent with what
// `tenex::detail::evaluate_impl` expects. Specifically, it checks that the
// canonical multi-indices of independent tensor components are ordered such
// that within each subset of symmetric indices, the index values are
// ascending from the rightmost index to the left, which is what `evaluate_impl`
// assumes.
void test_evaluate_and_canon_multi_index_consistency() {
  const std::string error_msg =
      "tenex::evaluate() assumes a canonical form for multi-indices that is no "
      "longer consistent with the canonical form defined by "
      "Tensor_detail::Structure. The logic of this unit test and "
      "tenex::detail::evaluate_impl must be updated to agree with the current "
      "canonical form for multi-indices.";
  INFO(error_msg);

  using datatype = double;
  using frame = Frame::Inertial;

  using iii = tnsr::iii<datatype, 3>::structure;
  using aaa = Tensor<datatype, Symmetry<1, 1, 1>,
                     index_list<SpacetimeIndex<3, UpLo::Lo, frame>,
                                SpacetimeIndex<3, UpLo::Lo, frame>,
                                SpacetimeIndex<3, UpLo::Lo, frame>>>::structure;
  using iaai = Tensor<datatype, Symmetry<1, 2, 2, 1>,
                      index_list<SpatialIndex<3, UpLo::Lo, frame>,
                                 SpacetimeIndex<3, UpLo::Lo, frame>,
                                 SpacetimeIndex<3, UpLo::Lo, frame>,
                                 SpatialIndex<3, UpLo::Lo, frame>>>::structure;
  using iiaa =
      Tensor<datatype, Symmetry<2, 2, 1, 1>,
             index_list<SpatialIndex<2, UpLo::Lo, frame>,
                        SpatialIndex<2, UpLo::Lo, frame>,
                        SpacetimeIndex<2, UpLo::Lo, frame>,
                        SpacetimeIndex<2, UpLo::Lo, frame>>>::structure;
  using aiai = Tensor<datatype, Symmetry<2, 1, 2, 1>,
                      index_list<SpacetimeIndex<3, UpLo::Lo, frame>,
                                 SpatialIndex<3, UpLo::Lo, frame>,
                                 SpacetimeIndex<3, UpLo::Lo, frame>,
                                 SpatialIndex<3, UpLo::Lo, frame>>>::structure;
  using iiii = Tensor<datatype, Symmetry<1, 1, 1, 1>,
                      index_list<SpatialIndex<3, UpLo::Lo, frame>,
                                 SpatialIndex<3, UpLo::Lo, frame>,
                                 SpatialIndex<3, UpLo::Lo, frame>,
                                 SpatialIndex<3, UpLo::Lo, frame>>>::structure;

  for (size_t i = 0; i < iii::size(); i++) {
    const auto canon_multi_index = iii::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[1]);
    CHECK(canon_multi_index[1] >= canon_multi_index[2]);
  }

  for (size_t i = 0; i < aaa::size(); i++) {
    const auto canon_multi_index = aaa::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[1]);
    CHECK(canon_multi_index[1] >= canon_multi_index[2]);
  }

  for (size_t i = 0; i < iaai::size(); i++) {
    const auto canon_multi_index = iaai::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[3]);
    CHECK(canon_multi_index[1] >= canon_multi_index[2]);
  }

  for (size_t i = 0; i < iiaa::size(); i++) {
    const auto canon_multi_index = iiaa::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[1]);
    CHECK(canon_multi_index[2] >= canon_multi_index[3]);
  }

  for (size_t i = 0; i < aiai::size(); i++) {
    const auto canon_multi_index = aiai::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[2]);
    CHECK(canon_multi_index[1] >= canon_multi_index[3]);
  }

  for (size_t i = 0; i < iiii::size(); i++) {
    const auto canon_multi_index = iiii::get_canonical_tensor_index(i);

    CHECK(canon_multi_index[0] >= canon_multi_index[1]);
    CHECK(canon_multi_index[1] >= canon_multi_index[2]);
    CHECK(canon_multi_index[2] >= canon_multi_index[3]);
  }
}

// \brief Test evaluation of rank 0, rank 1, and rank 2 tensors
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_evaluate_rank_012() {
  // Rank 0
  TestHelpers::tenex::test_evaluate_rank_0<DataType>();

  // Rank 1: spacetime
  TestHelpers::tenex::test_evaluate_rank_1<true, DataType,
                                           indextype_list<spacetime_index>,
                                           ti::a, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_1<
      true, DataType, indextype_list<spacetime_index>, ti::b, Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_1<true, DataType,
                                           indextype_list<spacetime_index>,
                                           ti::A, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_1<
      true, DataType, indextype_list<spacetime_index>, ti::B, Frame::Grid>();

  // Rank 1: spatial
  TestHelpers::tenex::test_evaluate_rank_1<
      true, DataType, indextype_list<spatial_index>, ti::i, Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_1<
      true, DataType, indextype_list<spatial_index>, ti::j, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_1<
      true, DataType, indextype_list<spatial_index>, ti::I, Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_1<
      true, DataType, indextype_list<spatial_index>, ti::J, Frame::Inertial>();

  // Rank 2: nonsymmetric, spacetime only
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spacetime_index, spacetime_index>, ti::a, ti::b,
      Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spacetime_index, spacetime_index>, ti::D, ti::C,
      Frame::Distorted>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spacetime_index, spacetime_index>, ti::e, ti::F,
      Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spacetime_index, spacetime_index>, ti::G, ti::b,
      Frame::ElementLogical>();

  // Rank 2: nonsymmetric, spatial only
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spatial_index, spatial_index>, ti::j, ti::i,
      Frame::ElementLogical>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spatial_index, spatial_index>, ti::I, ti::J,
      Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spatial_index, spatial_index>, ti::k, ti::M,
      Frame::Distorted>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spatial_index, spatial_index>, ti::M, ti::k,
      Frame::Inertial>();

  // Rank 2: nonsymmetric, spacetime and spatial mixed
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spacetime_index, spatial_index>, ti::c, ti::I,
      Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spacetime_index, spatial_index>, ti::A, ti::i,
      Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spatial_index, spacetime_index>, ti::J, ti::C,
      Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<2, 1>,
      indextype_list<spatial_index, spacetime_index>, ti::m, ti::e,
      Frame::Grid>();

  // Rank 2: symmetric, spacetime
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<1, 1>,
      indextype_list<spacetime_index, spacetime_index>, ti::a, ti::d,
      Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_2<
      false, DataType, Symmetry<1, 1>,
      indextype_list<spacetime_index, spacetime_index>, ti::a, ti::d,
      Frame::Inertial, Symmetry<2, 1>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<1, 1>,
      indextype_list<spacetime_index, spacetime_index>, ti::G, ti::B,
      Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_2<
      false, DataType, Symmetry<1, 1>,
      indextype_list<spacetime_index, spacetime_index>, ti::G, ti::B,
      Frame::Grid, Symmetry<2, 1>>();

  // Rank 2: symmetric, spatial
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<1, 1>,
      indextype_list<spatial_index, spatial_index>, ti::j, ti::i,
      Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_2<
      false, DataType, Symmetry<1, 1>,
      indextype_list<spatial_index, spatial_index>, ti::j, ti::i, Frame::Grid,
      Symmetry<2, 1>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, DataType, Symmetry<1, 1>,
      indextype_list<spatial_index, spatial_index>, ti::I, ti::J,
      Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_2<
      false, DataType, Symmetry<1, 1>,
      indextype_list<spatial_index, spatial_index>, ti::I, ti::J,
      Frame::Inertial, Symmetry<2, 1>>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Evaluate",
                  "[DataStructures][Unit]") {
  // Test tenex::evaluate implementation details
  test_contains_indices_to_contract();
  test_lhs_tensorindex_reorder_symm_consistency();
  test_evaluate_and_canon_multi_index_consistency();

  // Test tenex::evaluate for ranks 0, 1, and 2 tensors
  test_evaluate_rank_012<double>();
  test_evaluate_rank_012<DataVector>();
}
