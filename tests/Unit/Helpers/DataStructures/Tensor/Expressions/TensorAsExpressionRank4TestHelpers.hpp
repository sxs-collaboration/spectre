// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace TestHelpers::TensorExpressions {
/// \ingroup TestingFrameworkGroup
/// \brief Test that the transformation between LHS and RHS multi-indices and
/// the subsequent computed RHS multi-index of a rank 4 tensor is correctly
/// computed by the functions of TensorAsExpression, according to the orders of
/// the LHS and RHS generic indices
///
/// \details The functions tested are:
/// - `TensorAsExpression::compute_index_transformation`
/// - `TensorAsExpression::compute_rhs_multi_index`
///
/// If we consider the RHS tensor's generic indices to be (a, b, c, d), there
/// are 24 permutations that are possible orderings of the LHS tensor's generic
/// indices, such as: (a, b, c, d), (a, b, d, c), (a, c, b, d), etc. For each of
/// these cases, this test checks that for each LHS component's multi-index, the
/// equivalent RHS multi-index is correctly computed.
///
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \param tensorindex_c the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
/// \param tensorindex_d the fourth TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_D`
template <typename TensorIndexA, typename TensorIndexB, typename TensorIndexC,
          typename TensorIndexD>
void test_tensor_as_expression_rank_4(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c,
    const TensorIndexD& tensorindex_d) noexcept {
  const size_t dim_a = 4;
  const size_t dim_b = 2;
  const size_t dim_c = 3;
  const size_t dim_d = 1;

  const IndexType indextype_a =
      TensorIndexA::is_spacetime ? IndexType::Spacetime : IndexType::Spatial;
  const IndexType indextype_b =
      TensorIndexB::is_spacetime ? IndexType::Spacetime : IndexType::Spatial;
  const IndexType indextype_c =
      TensorIndexC::is_spacetime ? IndexType::Spacetime : IndexType::Spatial;
  const IndexType indextype_d =
      TensorIndexD::is_spacetime ? IndexType::Spacetime : IndexType::Spatial;

  Tensor<
      double, Symmetry<4, 3, 2, 1>,
      index_list<Tensor_detail::TensorIndexType<dim_a, TensorIndexA::valence,
                                                Frame::Inertial, indextype_a>,
                 Tensor_detail::TensorIndexType<dim_b, TensorIndexB::valence,
                                                Frame::Inertial, indextype_b>,
                 Tensor_detail::TensorIndexType<dim_c, TensorIndexC::valence,
                                                Frame::Inertial, indextype_c>,
                 Tensor_detail::TensorIndexType<dim_d, TensorIndexD::valence,
                                                Frame::Inertial, indextype_d>>>
      rhs_tensor{};
  std::iota(rhs_tensor.begin(), rhs_tensor.end(), 0.0);
  // Get TensorExpression from RHS tensor
  const auto R_abcd_expr =
      rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d);

  const std::array<size_t, 4> index_order_abcd = {
      TensorIndexA::value, TensorIndexB::value, TensorIndexC::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_abdc = {
      TensorIndexA::value, TensorIndexB::value, TensorIndexD::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_acbd = {
      TensorIndexA::value, TensorIndexC::value, TensorIndexB::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_acdb = {
      TensorIndexA::value, TensorIndexC::value, TensorIndexD::value,
      TensorIndexB::value};
  const std::array<size_t, 4> index_order_adbc = {
      TensorIndexA::value, TensorIndexD::value, TensorIndexB::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_adcb = {
      TensorIndexA::value, TensorIndexD::value, TensorIndexC::value,
      TensorIndexB::value};

  const std::array<size_t, 4> index_order_bacd = {
      TensorIndexB::value, TensorIndexA::value, TensorIndexC::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_badc = {
      TensorIndexB::value, TensorIndexA::value, TensorIndexD::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_bcad = {
      TensorIndexB::value, TensorIndexC::value, TensorIndexA::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_bcda = {
      TensorIndexB::value, TensorIndexC::value, TensorIndexD::value,
      TensorIndexA::value};
  const std::array<size_t, 4> index_order_bdac = {
      TensorIndexB::value, TensorIndexD::value, TensorIndexA::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_bdca = {
      TensorIndexB::value, TensorIndexD::value, TensorIndexC::value,
      TensorIndexA::value};

  const std::array<size_t, 4> index_order_cabd = {
      TensorIndexC::value, TensorIndexA::value, TensorIndexB::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_cadb = {
      TensorIndexC::value, TensorIndexA::value, TensorIndexD::value,
      TensorIndexB::value};
  const std::array<size_t, 4> index_order_cbad = {
      TensorIndexC::value, TensorIndexB::value, TensorIndexA::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_cbda = {
      TensorIndexC::value, TensorIndexB::value, TensorIndexD::value,
      TensorIndexA::value};
  const std::array<size_t, 4> index_order_cdab = {
      TensorIndexC::value, TensorIndexD::value, TensorIndexA::value,
      TensorIndexB::value};
  const std::array<size_t, 4> index_order_cdba = {
      TensorIndexC::value, TensorIndexD::value, TensorIndexB::value,
      TensorIndexA::value};

  const std::array<size_t, 4> index_order_dabc = {
      TensorIndexD::value, TensorIndexA::value, TensorIndexB::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_dacb = {
      TensorIndexD::value, TensorIndexA::value, TensorIndexC::value,
      TensorIndexB::value};
  const std::array<size_t, 4> index_order_dbac = {
      TensorIndexD::value, TensorIndexB::value, TensorIndexA::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_dbca = {
      TensorIndexD::value, TensorIndexB::value, TensorIndexC::value,
      TensorIndexA::value};
  const std::array<size_t, 4> index_order_dcab = {
      TensorIndexD::value, TensorIndexC::value, TensorIndexA::value,
      TensorIndexB::value};
  const std::array<size_t, 4> index_order_dcba = {
      TensorIndexD::value, TensorIndexC::value, TensorIndexB::value,
      TensorIndexA::value};

  const std::array<size_t, 4> actual_abcd_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_abcd);
  const std::array<size_t, 4> expected_abcd_to_abcd_transformation = {0, 1, 2,
                                                                      3};
  const std::array<size_t, 4> actual_abdc_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_abdc);
  const std::array<size_t, 4> expected_abdc_to_abcd_transformation = {0, 1, 3,
                                                                      2};
  const std::array<size_t, 4> actual_acbd_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_acbd);
  const std::array<size_t, 4> expected_acbd_to_abcd_transformation = {0, 2, 1,
                                                                      3};
  const std::array<size_t, 4> actual_acdb_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_acdb);
  const std::array<size_t, 4> expected_acdb_to_abcd_transformation = {0, 3, 1,
                                                                      2};
  const std::array<size_t, 4> actual_adbc_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_adbc);
  const std::array<size_t, 4> expected_adbc_to_abcd_transformation = {0, 2, 3,
                                                                      1};
  const std::array<size_t, 4> actual_adcb_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_adcb);
  const std::array<size_t, 4> expected_adcb_to_abcd_transformation = {0, 3, 2,
                                                                      1};

  const std::array<size_t, 4> actual_bacd_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_bacd);
  const std::array<size_t, 4> expected_bacd_to_abcd_transformation = {1, 0, 2,
                                                                      3};
  const std::array<size_t, 4> actual_badc_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_badc);
  const std::array<size_t, 4> expected_badc_to_abcd_transformation = {1, 0, 3,
                                                                      2};
  const std::array<size_t, 4> actual_bcad_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_bcad);
  const std::array<size_t, 4> expected_bcad_to_abcd_transformation = {2, 0, 1,
                                                                      3};
  const std::array<size_t, 4> actual_bcda_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_bcda);
  const std::array<size_t, 4> expected_bcda_to_abcd_transformation = {3, 0, 1,
                                                                      2};
  const std::array<size_t, 4> actual_bdac_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_bdac);
  const std::array<size_t, 4> expected_bdac_to_abcd_transformation = {2, 0, 3,
                                                                      1};
  const std::array<size_t, 4> actual_bdca_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_bdca);
  const std::array<size_t, 4> expected_bdca_to_abcd_transformation = {3, 0, 2,
                                                                      1};

  const std::array<size_t, 4> actual_cabd_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_cabd);
  const std::array<size_t, 4> expected_cabd_to_abcd_transformation = {1, 2, 0,
                                                                      3};
  const std::array<size_t, 4> actual_cadb_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_cadb);
  const std::array<size_t, 4> expected_cadb_to_abcd_transformation = {1, 3, 0,
                                                                      2};
  const std::array<size_t, 4> actual_cbad_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_cbad);
  const std::array<size_t, 4> expected_cbad_to_abcd_transformation = {2, 1, 0,
                                                                      3};
  const std::array<size_t, 4> actual_cbda_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_cbda);
  const std::array<size_t, 4> expected_cbda_to_abcd_transformation = {3, 1, 0,
                                                                      2};
  const std::array<size_t, 4> actual_cdab_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_cdab);
  const std::array<size_t, 4> expected_cdab_to_abcd_transformation = {2, 3, 0,
                                                                      1};
  const std::array<size_t, 4> actual_cdba_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_cdba);
  const std::array<size_t, 4> expected_cdba_to_abcd_transformation = {3, 2, 0,
                                                                      1};

  const std::array<size_t, 4> actual_dabc_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_dabc);
  const std::array<size_t, 4> expected_dabc_to_abcd_transformation = {1, 2, 3,
                                                                      0};
  const std::array<size_t, 4> actual_dacb_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_dacb);
  const std::array<size_t, 4> expected_dacb_to_abcd_transformation = {1, 3, 2,
                                                                      0};
  const std::array<size_t, 4> actual_dbac_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_dbac);
  const std::array<size_t, 4> expected_dbac_to_abcd_transformation = {2, 1, 3,
                                                                      0};
  const std::array<size_t, 4> actual_dbca_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_dbca);
  const std::array<size_t, 4> expected_dbca_to_abcd_transformation = {3, 1, 2,
                                                                      0};
  const std::array<size_t, 4> actual_dcab_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_dcab);
  const std::array<size_t, 4> expected_dcab_to_abcd_transformation = {2, 3, 1,
                                                                      0};
  const std::array<size_t, 4> actual_dcba_to_abcd_transformation =
      R_abcd_expr.compute_index_transformation(index_order_dcba);
  const std::array<size_t, 4> expected_dcba_to_abcd_transformation = {3, 2, 1,
                                                                      0};

  CHECK(actual_abcd_to_abcd_transformation ==
        expected_abcd_to_abcd_transformation);
  CHECK(actual_abdc_to_abcd_transformation ==
        expected_abdc_to_abcd_transformation);
  CHECK(actual_acbd_to_abcd_transformation ==
        expected_acbd_to_abcd_transformation);
  CHECK(actual_acdb_to_abcd_transformation ==
        expected_acdb_to_abcd_transformation);
  CHECK(actual_adbc_to_abcd_transformation ==
        expected_adbc_to_abcd_transformation);
  CHECK(actual_adcb_to_abcd_transformation ==
        expected_adcb_to_abcd_transformation);

  CHECK(actual_bacd_to_abcd_transformation ==
        expected_bacd_to_abcd_transformation);
  CHECK(actual_badc_to_abcd_transformation ==
        expected_badc_to_abcd_transformation);
  CHECK(actual_bcad_to_abcd_transformation ==
        expected_bcad_to_abcd_transformation);
  CHECK(actual_bcda_to_abcd_transformation ==
        expected_bcda_to_abcd_transformation);
  CHECK(actual_bdac_to_abcd_transformation ==
        expected_bdac_to_abcd_transformation);
  CHECK(actual_bdca_to_abcd_transformation ==
        expected_bdca_to_abcd_transformation);

  CHECK(actual_cabd_to_abcd_transformation ==
        expected_cabd_to_abcd_transformation);
  CHECK(actual_cadb_to_abcd_transformation ==
        expected_cadb_to_abcd_transformation);
  CHECK(actual_cbad_to_abcd_transformation ==
        expected_cbad_to_abcd_transformation);
  CHECK(actual_cbda_to_abcd_transformation ==
        expected_cbda_to_abcd_transformation);
  CHECK(actual_cdab_to_abcd_transformation ==
        expected_cdab_to_abcd_transformation);
  CHECK(actual_cdba_to_abcd_transformation ==
        expected_cdba_to_abcd_transformation);

  CHECK(actual_dabc_to_abcd_transformation ==
        expected_dabc_to_abcd_transformation);
  CHECK(actual_dacb_to_abcd_transformation ==
        expected_dacb_to_abcd_transformation);
  CHECK(actual_dbac_to_abcd_transformation ==
        expected_dbac_to_abcd_transformation);
  CHECK(actual_dbca_to_abcd_transformation ==
        expected_dbca_to_abcd_transformation);
  CHECK(actual_dcab_to_abcd_transformation ==
        expected_dcab_to_abcd_transformation);
  CHECK(actual_dcba_to_abcd_transformation ==
        expected_dcba_to_abcd_transformation);

  for (size_t i = 0; i < dim_a; i++) {
    for (size_t j = 0; j < dim_b; j++) {
      for (size_t k = 0; k < dim_c; k++) {
        for (size_t l = 0; l < dim_d; l++) {
          const std::array<size_t, 4> ijkl = {i, j, k, l};
          const std::array<size_t, 4> ijlk = {i, j, l, k};
          const std::array<size_t, 4> ikjl = {i, k, j, l};
          const std::array<size_t, 4> iklj = {i, k, l, j};
          const std::array<size_t, 4> iljk = {i, l, j, k};
          const std::array<size_t, 4> ilkj = {i, l, k, j};

          const std::array<size_t, 4> jikl = {j, i, k, l};
          const std::array<size_t, 4> jilk = {j, i, l, k};
          const std::array<size_t, 4> jkil = {j, k, i, l};
          const std::array<size_t, 4> jkli = {j, k, l, i};
          const std::array<size_t, 4> jlik = {j, l, i, k};
          const std::array<size_t, 4> jlki = {j, l, k, i};

          const std::array<size_t, 4> kijl = {k, i, j, l};
          const std::array<size_t, 4> kilj = {k, i, l, j};
          const std::array<size_t, 4> kjil = {k, j, i, l};
          const std::array<size_t, 4> kjli = {k, j, l, i};
          const std::array<size_t, 4> klij = {k, l, i, j};
          const std::array<size_t, 4> klji = {k, l, j, i};

          const std::array<size_t, 4> lijk = {l, i, j, k};
          const std::array<size_t, 4> likj = {l, i, k, j};
          const std::array<size_t, 4> ljik = {l, j, i, k};
          const std::array<size_t, 4> ljki = {l, j, k, i};
          const std::array<size_t, 4> lkij = {l, k, i, j};
          const std::array<size_t, 4> lkji = {l, k, j, i};

          // For L_{abcd} = R_{abcd}, check that L_{ijkl} == R_{ijkl}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_abcd_to_abcd_transformation) == ijkl);
          // For L_{abdc} = R_{abcd}, check that L_{ijkl} == R_{ijlk}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_abdc_to_abcd_transformation) == ijlk);
          // For L_{acbd} = R_{abcd}, check that L_{ijkl} == R_{ikjl}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_acbd_to_abcd_transformation) == ikjl);
          // For L_{acdb} = R_{abcd}, check that L_{ijkl} == R_{iljk}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_acdb_to_abcd_transformation) == iljk);
          // For L_{adbc} = R_{abcd}, check that L_{ijkl} == R_{iklj}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_adbc_to_abcd_transformation) == iklj);
          // For L_{adcb} = R_{abcd}, check that L_{ijkl} == R_{ilkj}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_adcb_to_abcd_transformation) == ilkj);

          // For L_{bacd} = R_{abcd}, check that L_{ijkl} == R_{jikl}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_bacd_to_abcd_transformation) == jikl);
          // For L_{badc} = R_{abcd}, check that L_{ijkl} == R_{jilk}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_badc_to_abcd_transformation) == jilk);
          // For L_{bcad} = R_{abcd}, check that L_{ijkl} == R_{kijl}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_bcad_to_abcd_transformation) == kijl);
          // For L_{bcda} = R_{abcd}, check that L_{ijkl} == R_{lijk}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_bcda_to_abcd_transformation) == lijk);
          // For L_{bdac} = R_{abcd}, check that L_{ijkl} == R_{kilj}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_bdac_to_abcd_transformation) == kilj);
          // For L_{bdca} = R_{abcd}, check that L_{ijkl} == R_{likj}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_bdca_to_abcd_transformation) == likj);

          // For L_{cabd} = R_{abcd}, check that L_{ijkl} == R_{jkil}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_cabd_to_abcd_transformation) == jkil);
          // For L_{cadb} = R_{abcd}, check that L_{ijkl} == R_{jlik}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_cadb_to_abcd_transformation) == jlik);
          // For L_{cbad} = R_{abcd}, check that L_{ijkl} == R_{kjil}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_cbad_to_abcd_transformation) == kjil);
          // For L_{cbda} = R_{abcd}, check that L_{ijkl} == R_{ljik}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_cbda_to_abcd_transformation) == ljik);
          // For L_{cdab} = R_{abcd}, check that L_{ijkl} == R_{klij}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_cdab_to_abcd_transformation) == klij);
          // For L_{cdba} = R_{abcd}, check that L_{ijkl} == R_{lkij}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_cdba_to_abcd_transformation) == lkij);

          // For L_{dabc} = R_{abcd}, check that L_{ijkl} == R_{jkli}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_dabc_to_abcd_transformation) == jkli);
          // For L_{dacb} = R_{abcd}, check that L_{ijkl} == R_{jlki}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_dacb_to_abcd_transformation) == jlki);
          // For L_{dbac} = R_{abcd}, check that L_{ijkl} == R_{kjli}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_dbac_to_abcd_transformation) == kjli);
          // For L_{dbca} = R_{abcd}, check that L_{ijkl} == R_{ljki}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_dbca_to_abcd_transformation) == ljki);
          // For L_{dcab} = R_{abcd}, check that L_{ijkl} == R_{klji}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_dcab_to_abcd_transformation) == klji);
          // For L_{dcba} = R_{abcd}, check that L_{ijkl} == R_{lkji}
          CHECK(R_abcd_expr.compute_rhs_multi_index(
                    ijkl, expected_dcba_to_abcd_transformation) == lkji);
        }
      }
    }
  }
}
}  // namespace TestHelpers::TensorExpressions
