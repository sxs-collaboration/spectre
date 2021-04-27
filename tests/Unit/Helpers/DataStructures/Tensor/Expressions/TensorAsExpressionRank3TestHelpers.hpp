// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace TestHelpers::TensorExpressions {
/// \ingroup TestingFrameworkGroup
/// \brief Test that the transformation between LHS and RHS multi-indices and
/// the subsequent computed RHS multi-index of a rank 3 tensor is correctly
/// computed by the functions of TensorAsExpression, according to the orders of
/// the LHS and RHS generic indices
///
/// \details The functions tested are:
/// - `TensorAsExpression::compute_index_transformation`
/// - `TensorAsExpression::compute_rhs_multi_index`
///
/// If we consider the RHS tensor's generic indices to be (a, b, c), the
/// possible orderings of the LHS tensor's generic indices are: (a, b, c),
/// (a, c, b), (b, a, c), (b, c, a), (c, a, b), and (c, b, a). For each of these
/// cases, this test checks that for each LHS component's multi-index, the
/// equivalent RHS multi-index is correctly computed.
///
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \param tensorindex_c the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
template <typename TensorIndexA, typename TensorIndexB, typename TensorIndexC>
void test_tensor_as_expression_rank_3(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
  const size_t dim_a = 4;
  const size_t dim_b = 2;
  const size_t dim_c = 3;

  const IndexType indextype_a =
      TensorIndexA::is_spacetime ? IndexType::Spacetime : IndexType::Spatial;
  const IndexType indextype_b =
      TensorIndexB::is_spacetime ? IndexType::Spacetime : IndexType::Spatial;
  const IndexType indextype_c =
      TensorIndexC::is_spacetime ? IndexType::Spacetime : IndexType::Spatial;

  Tensor<double, Symmetry<3, 2, 1>,
         index_list<Tensor_detail::TensorIndexType<dim_a, TensorIndexA::valence,
                                                   Frame::Grid, indextype_a>,
                    Tensor_detail::TensorIndexType<dim_b, TensorIndexB::valence,
                                                   Frame::Grid, indextype_b>,
                    Tensor_detail::TensorIndexType<dim_c, TensorIndexC::valence,
                                                   Frame::Grid, indextype_c>>>
      rhs_tensor{};
  std::iota(rhs_tensor.begin(), rhs_tensor.end(), 0.0);
  // Get TensorExpression from RHS tensor
  const auto R_abc_expr =
      rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c);

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
      R_abc_expr.compute_index_transformation(index_order_abc);
  const std::array<size_t, 3> expected_abc_to_abc_transformation = {0, 1, 2};
  const std::array<size_t, 3> actual_acb_to_abc_transformation =
      R_abc_expr.compute_index_transformation(index_order_acb);
  const std::array<size_t, 3> expected_acb_to_abc_transformation = {0, 2, 1};
  const std::array<size_t, 3> actual_bac_to_abc_transformation =
      R_abc_expr.compute_index_transformation(index_order_bac);
  const std::array<size_t, 3> expected_bac_to_abc_transformation = {1, 0, 2};
  const std::array<size_t, 3> actual_bca_to_abc_transformation =
      R_abc_expr.compute_index_transformation(index_order_bca);
  const std::array<size_t, 3> expected_bca_to_abc_transformation = {2, 0, 1};
  const std::array<size_t, 3> actual_cab_to_abc_transformation =
      R_abc_expr.compute_index_transformation(index_order_cab);
  const std::array<size_t, 3> expected_cab_to_abc_transformation = {1, 2, 0};
  const std::array<size_t, 3> actual_cba_to_abc_transformation =
      R_abc_expr.compute_index_transformation(index_order_cba);
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
        CHECK(R_abc_expr.compute_rhs_multi_index(
                  ijk, expected_abc_to_abc_transformation) == ijk);
        // For L_{acb} = R_{abc}, check that L_{ijk} == R_{ikj}
        CHECK(R_abc_expr.compute_rhs_multi_index(
                  ijk, expected_acb_to_abc_transformation) == ikj);
        // For L_{bac} = R_{abc}, check that L_{ijk} == R_{jik}
        CHECK(R_abc_expr.compute_rhs_multi_index(
                  ijk, expected_bac_to_abc_transformation) == jik);
        // For L_{bca} = R_{abc}, check that L_{ijk} == R_{kij}
        CHECK(R_abc_expr.compute_rhs_multi_index(
                  ijk, expected_bca_to_abc_transformation) == kij);
        // For L_{cab} = R_{abc}, check that L_{ijk} == R_{jki}
        CHECK(R_abc_expr.compute_rhs_multi_index(
                  ijk, expected_cab_to_abc_transformation) == jki);
        // For L_{cba} = R_{abc}, check that L_{ijk} == R_{kji}
        CHECK(R_abc_expr.compute_rhs_multi_index(
                  ijk, expected_cba_to_abc_transformation) == kji);
      }
    }
  }
}
}  // namespace TestHelpers::TensorExpressions
