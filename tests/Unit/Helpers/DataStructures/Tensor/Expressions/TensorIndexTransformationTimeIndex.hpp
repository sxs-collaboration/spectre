// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndexTransformation.hpp"

namespace TestHelpers::TensorExpressions {
/// \ingroup TestingFrameworkGroup
/// \brief Test that the transformation between two tensors generic
/// indices and the subsequent transformed multi-indices are correctly computed
/// when time indices are used with at least one of the tensors
///
/// \details The functions tested are:
/// - `TensorExpressions::compute_tensorindex_transformation`
/// - `TensorExpressions::transform_multi_index`
void test_tensor_index_transformation_with_time_indices() noexcept {
  const std::array<size_t, 0> index_order_empty = {};
  const std::array<size_t, 1> index_order_t = {ti_t.value};
  const std::array<size_t, 3> index_order_atb = {ti_a.value, ti_t.value,
                                                 ti_b.value};
  const std::array<size_t, 3> index_order_Ttt = {ti_T.value, ti_t.value,
                                                 ti_t.value};
  const std::array<size_t, 5> index_order_tbatT = {
      ti_t.value, ti_b.value, ti_a.value, ti_t.value, ti_T.value};

  const size_t time_index_placeholder = ::TensorExpressions::
      TensorIndexTransformation_detail::time_index_position_placeholder;

  // Transform multi-index for rank 0 tensor to multi-index for rank 1 tensor
  // with time index
  // e.g. transform multi-index for L to multi-index for R_{t}
  const std::array<size_t, 1> actual_empty_to_t_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order_empty,
                                                              index_order_t);
  const std::array<size_t, 1> expected_empty_to_t_transformation = {
      time_index_placeholder};
  CHECK(actual_empty_to_t_transformation == expected_empty_to_t_transformation);
  const std::array<size_t, 0> multi_index_empty = {};
  const std::array<size_t, 1> multi_index_t = {0};
  CHECK(::TensorExpressions::transform_multi_index(
            multi_index_empty, expected_empty_to_t_transformation) ==
        multi_index_t);

  // Transform multi-index for rank 1 tensor with time index to multi-index for
  // rank 0 tensor
  // e.g. transform multi-index for L_{t} to multi-index for R
  const std::array<size_t, 0> actual_t_to_empty_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(
          index_order_t, index_order_empty);
  const std::array<size_t, 0> expected_t_to_empty_transformation = {};
  CHECK(actual_t_to_empty_transformation == expected_t_to_empty_transformation);
  CHECK(::TensorExpressions::transform_multi_index(
            multi_index_t, expected_t_to_empty_transformation) ==
        multi_index_empty);

  // Transform multi-index for rank 0 tensor to multi-index for rank 3 tensor
  // with three time indices
  // e.g. transform multi-index for L to multi-index for R^{t}{}_{tt}
  const std::array<size_t, 3> actual_empty_to_Ttt_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order_empty,
                                                              index_order_Ttt);
  const std::array<size_t, 3> expected_empty_to_Ttt_transformation = {
      time_index_placeholder, time_index_placeholder, time_index_placeholder};
  CHECK(actual_empty_to_Ttt_transformation ==
        expected_empty_to_Ttt_transformation);
  const std::array<size_t, 3> multi_index_Ttt = {0, 0, 0};
  CHECK(::TensorExpressions::transform_multi_index(
            multi_index_empty, expected_empty_to_Ttt_transformation) ==
        multi_index_Ttt);

  // Transform multi-index for rank 3 tensor with three time indices to
  // multi-index for rank 0 tensor
  // e.g. transform multi-index for L^{t}{}_{tt} to multi-index for R
  const std::array<size_t, 0> actual_Ttt_to_empty_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(
          index_order_Ttt, index_order_empty);
  const std::array<size_t, 0> expected_Ttt_to_empty_transformation = {};
  CHECK(actual_Ttt_to_empty_transformation ==
        expected_Ttt_to_empty_transformation);
  CHECK(::TensorExpressions::transform_multi_index(
            multi_index_Ttt, expected_Ttt_to_empty_transformation) ==
        multi_index_empty);

  // Transform multi-indices for rank 3 tensor with one time index to
  // multi-indices for rank 5 tensor with three time indices
  // e.g. transform multi-indices for L_{atb} to multi-indices for
  // R_{tbat}{}^{t}
  //
  // Note: While SpECTRE Tensors only support up to rank 4, TensorExpressions
  // may represent higher rank intermediate expressions in an equation. In
  // addition, this case means to get at testing having a different number of
  // time indices on either side in addition to a different relative index order
  // for generic indices (i.e. a and b)
  const std::array<size_t, 5> actual_atb_to_tbatT_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(
          index_order_atb, index_order_tbatT);
  const std::array<size_t, 5> expected_atb_to_tbatT_transformation = {
      time_index_placeholder, 2, 0, time_index_placeholder,
      time_index_placeholder};
  CHECK(actual_atb_to_tbatT_transformation ==
        expected_atb_to_tbatT_transformation);

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      const std::array<size_t, 3> multi_index_atb = {a, 0, b};
      const std::array<size_t, 5> multi_index_tbatT = {0, b, a, 0, 0};
      CHECK(::TensorExpressions::transform_multi_index(
                multi_index_atb, expected_atb_to_tbatT_transformation) ==
            multi_index_tbatT);
    }
  }

  // Transform multi-indices for rank 5 tensor with three time indices to
  // multi-indices for rank 3 tensor with one time index time index
  // e.g. transform multi-indices for L_{tbat}{}^{t} to multi-indices for
  // R_{atb}
  const std::array<size_t, 3> actual_tbatT_to_atb_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order_tbatT,
                                                              index_order_atb);
  const std::array<size_t, 3> expected_tbatT_to_atb_transformation = {
      2, time_index_placeholder, 1};
  CHECK(actual_tbatT_to_atb_transformation ==
        expected_tbatT_to_atb_transformation);

  for (size_t b = 0; b < 4; b++) {
    for (size_t a = 0; a < 4; a++) {
      const std::array<size_t, 5> multi_index_tbatT = {0, b, a, 0, 0};
      const std::array<size_t, 3> multi_index_atb = {a, 0, b};
      CHECK(::TensorExpressions::transform_multi_index(
                multi_index_tbatT, expected_tbatT_to_atb_transformation) ==
            multi_index_atb);
    }
  }
}
}  // namespace TestHelpers::TensorExpressions
