// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/TensorIndexTransformation.hpp"

namespace TestHelpers::TensorExpressions {
/// \ingroup TestingFrameworkGroup
/// \brief Test that the transformation between two rank 3 tensors' generic
/// indices and the subsequent transformed multi-indices are correctly computed
///
/// \details The functions tested are:
/// - `TensorExpressions::compute_tensorindex_transformation`
/// - `TensorExpressions::transform_multi_index`
///
/// If we consider the first tensor's generic indices to be (a, b, c), the
/// possible orderings of the second tensor's generic indices are: (a, b, c),
/// (a, c, b), (b, a, c), (b, c, a), (c, a, b), and (c, b, a). For each of
/// these cases, this test checks that for each multi-index with the first
/// generic index ordering, the equivalent multi-index with the second ordering
/// is correctly computed.
///
/// \tparam TensorIndexA the first generic tensor index, e.g. type of `ti_a`
/// \tparam TensorIndexB the second generic tensor index, e.g. type of `ti_B`
/// \tparam TensorIndexC the third generic tensor index, e.g. type of `ti_c`
template <typename TensorIndexA, typename TensorIndexB, typename TensorIndexC>
void test_tensor_index_transformation_rank_3(
    const TensorIndexA& /*tensorindex_a*/,
    const TensorIndexB& /*tensorindex_b*/,
    const TensorIndexC& /*tensorindex_c*/) noexcept {
  const size_t dim_a = 4;
  const size_t dim_b = 2;
  const size_t dim_c = 3;

  const std::array<size_t, 3> index_order_abc = {
      TensorIndexA::value, TensorIndexB::value, TensorIndexC::value};
  const std::array<size_t, 3> index_order_acb = {
      TensorIndexA::value, TensorIndexC::value, TensorIndexB::value};
  const std::array<size_t, 3> index_order_bac = {
      TensorIndexB::value, TensorIndexA::value, TensorIndexC::value};
  const std::array<size_t, 3> index_order_bca = {
      TensorIndexB::value, TensorIndexC::value, TensorIndexA::value};
  const std::array<size_t, 3> index_order_cab = {
      TensorIndexC::value, TensorIndexA::value, TensorIndexB::value};
  const std::array<size_t, 3> index_order_cba = {
      TensorIndexC::value, TensorIndexB::value, TensorIndexA::value};

  const std::array<size_t, 3> actual_abc_to_abc_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order_abc,
                                                              index_order_abc);
  const std::array<size_t, 3> expected_abc_to_abc_transformation = {0, 1, 2};
  const std::array<size_t, 3> actual_acb_to_abc_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order_acb,
                                                              index_order_abc);
  const std::array<size_t, 3> expected_acb_to_abc_transformation = {0, 2, 1};
  const std::array<size_t, 3> actual_bac_to_abc_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order_bac,
                                                              index_order_abc);
  const std::array<size_t, 3> expected_bac_to_abc_transformation = {1, 0, 2};
  const std::array<size_t, 3> actual_bca_to_abc_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order_bca,
                                                              index_order_abc);
  const std::array<size_t, 3> expected_bca_to_abc_transformation = {2, 0, 1};
  const std::array<size_t, 3> actual_cab_to_abc_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order_cab,
                                                              index_order_abc);
  const std::array<size_t, 3> expected_cab_to_abc_transformation = {1, 2, 0};
  const std::array<size_t, 3> actual_cba_to_abc_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order_cba,
                                                              index_order_abc);
  const std::array<size_t, 3> expected_cba_to_abc_transformation = {2, 1, 0};

  CHECK(actual_abc_to_abc_transformation == expected_abc_to_abc_transformation);
  CHECK(actual_acb_to_abc_transformation == expected_acb_to_abc_transformation);
  CHECK(actual_bac_to_abc_transformation == expected_bac_to_abc_transformation);
  CHECK(actual_bca_to_abc_transformation == expected_bca_to_abc_transformation);
  CHECK(actual_cab_to_abc_transformation == expected_cab_to_abc_transformation);
  CHECK(actual_cba_to_abc_transformation == expected_cba_to_abc_transformation);

  for (size_t i = 0; i < dim_a; i++) {
    for (size_t j = 0; j < dim_b; j++) {
      for (size_t k = 0; k < dim_c; k++) {
        const std::array<size_t, 3> ijk = {i, j, k};
        const std::array<size_t, 3> ikj = {i, k, j};
        const std::array<size_t, 3> jik = {j, i, k};
        const std::array<size_t, 3> jki = {j, k, i};
        const std::array<size_t, 3> kij = {k, i, j};
        const std::array<size_t, 3> kji = {k, j, i};

        // For L_{abc} = R_{abc}, check that L_{ijk} == R_{ijk}
        CHECK(::TensorExpressions::transform_multi_index(
                  ijk, expected_abc_to_abc_transformation) == ijk);
        // For L_{acb} = R_{abc}, check that L_{ijk} == R_{ikj}
        CHECK(::TensorExpressions::transform_multi_index(
                  ijk, expected_acb_to_abc_transformation) == ikj);
        // For L_{bac} = R_{abc}, check that L_{ijk} == R_{jik}
        CHECK(::TensorExpressions::transform_multi_index(
                  ijk, expected_bac_to_abc_transformation) == jik);
        // For L_{bca} = R_{abc}, check that L_{ijk} == R_{kij}
        CHECK(::TensorExpressions::transform_multi_index(
                  ijk, expected_bca_to_abc_transformation) == kij);
        // For L_{cab} = R_{abc}, check that L_{ijk} == R_{jki}
        CHECK(::TensorExpressions::transform_multi_index(
                  ijk, expected_cab_to_abc_transformation) == jki);
        // For L_{cba} = R_{abc}, check that L_{ijk} == R_{kji}
        CHECK(::TensorExpressions::transform_multi_index(
                  ijk, expected_cba_to_abc_transformation) == kji);
      }
    }
  }
}
}  // namespace TestHelpers::TensorExpressions
