// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once


#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::TensorExpressions {

/// \ingroup TestingFrameworkGroup
/// \brief Test that the computed tensor multi-index of a rank 4 RHS Tensor is
/// equivalent to the given LHS tensor multi-index, according to the order of
/// their generic indices
///
/// \details `TensorIndexA`, `TensorIndexB`, `TensorIndexC`, and  `TensorIndexD`
/// can be any type of TensorIndex and are not necessarily `ti_a_t`, `ti_b_t`,
/// `ti_c_t`, and `ti_d_t`. The "A", "B", "C", and "D" suffixes just denote the
/// ordering of the generic indices of the RHS tensor expression. In the RHS
/// tensor expression, it means `TensorIndexA` is the first index used,
/// `TensorIndexB` is the second index used, `TensorIndexC` is the third index
/// used, and `TensorIndexD` is the fourth index used.
///
/// If we consider the RHS tensor's generic indices to be (a, b, c, d), there
/// are 24 permutations that are possible orderings of the LHS tensor's generic
/// indices, such as: (a, b, c, d), e.g. (a, b, d, c), (a, c, b, d), etc. For
/// each of these cases, this test checks that for each LHS component's tensor
/// multi-index, the equivalent RHS tensor multi-index is correctly computed.
///
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \param tensorindex_c the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
/// \param tensorindex_d the fourth TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_D`
template <typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC, typename TensorIndexD>
void test_compute_rhs_tensor_index_rank_4(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c,
    const TensorIndexD& tensorindex_d) noexcept {
  const Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList> rhs_tensor(5_st);
  // Get TensorExpression from RHS tensor
  const auto R_abcd =
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

  const size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<RhsTensorIndexTypeList, 2>::dim;
  const size_t dim_d = tmpl::at_c<RhsTensorIndexTypeList, 3>::dim;

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
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_abcd, index_order_abcd, ijkl) == ijkl);
          // For L_{abdc} = R_{abcd}, check that L_{ijkl} == R_{ijlk}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_abdc, index_order_abcd, ijkl) == ijlk);
          // For L_{acbd} = R_{abcd}, check that L_{ijkl} == R_{ikjl}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_acbd, index_order_abcd, ijkl) == ikjl);
          // For L_{acdb} = R_{abcd}, check that L_{ijkl} == R_{iklj}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_acdb, index_order_abcd, ijkl) == iljk);
          // For L_{adbc} = R_{abcd}, check that L_{ijkl} == R_{iklj}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_adbc, index_order_abcd, ijkl) == iklj);
          // For L_{adcb} = R_{abcd}, check that L_{ijkl} == R_{ilkj}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_adcb, index_order_abcd, ijkl) == ilkj);

          // For L_{bacd} = R_{abcd}, check that L_{ijkl} == R_{jikl}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_bacd, index_order_abcd, ijkl) == jikl);
          // For L_{badc} = R_{abcd}, check that L_{ijkl} == R_{jilk}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_badc, index_order_abcd, ijkl) == jilk);
          // For L_{bcad} = R_{abcd}, check that L_{ijkl} == R_{kijl}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_bcad, index_order_abcd, ijkl) == kijl);
          // For L_{bcda} = R_{abcd}, check that L_{ijkl} == R_{lijk}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_bcda, index_order_abcd, ijkl) == lijk);
          // For L_{bdac} = R_{abcd}, check that L_{ijkl} == R_{kilj}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_bdac, index_order_abcd, ijkl) == kilj);
          // For L_{bdca} = R_{abcd}, check that L_{ijkl} == R_{likj}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_bdca, index_order_abcd, ijkl) == likj);

          // For L_{cabd} = R_{abcd}, check that L_{ijkl} == R_{jkil}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_cabd, index_order_abcd, ijkl) == jkil);
          // For L_{cadb} = R_{abcd}, check that L_{ijkl} == R_{jlik}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_cadb, index_order_abcd, ijkl) == jlik);
          // For L_{cbad} = R_{abcd}, check that L_{ijkl} == R_{kjil}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_cbad, index_order_abcd, ijkl) == kjil);
          // For L_{cbda} = R_{abcd}, check that L_{ijkl} == R_{ljik}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_cbda, index_order_abcd, ijkl) == ljik);
          // For L_{cdab} = R_{abcd}, check that L_{ijkl} == R_{klij}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_cdab, index_order_abcd, ijkl) == klij);
          // For L_{cdba} = R_{abcd}, check that L_{ijkl} == R_{lkij}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_cdba, index_order_abcd, ijkl) == lkij);

          // For L_{dabc} = R_{abcd}, check that L_{ijkl} == R_{jkli}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_dabc, index_order_abcd, ijkl) == jkli);
          // For L_{dacb} = R_{abcd}, check that L_{ijkl} == R_{jlki}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_dacb, index_order_abcd, ijkl) == jlki);
          // For L_{dbac} = R_{abcd}, check that L_{ijkl} == R_{kjli}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_dbac, index_order_abcd, ijkl) == kjli);
          // For L_{dbca} = R_{abcd}, check that L_{ijkl} == R_{ljki}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_dbca, index_order_abcd, ijkl) == ljki);
          // For L_{dcab} = R_{abcd}, check that L_{ijkl} == R_{klji}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_dcab, index_order_abcd, ijkl) == klji);
          // For L_{dcba} = R_{abcd}, check that L_{ijkl} == R_{lkji}
          CHECK(R_abcd.compute_rhs_tensor_index(
                    index_order_dcba, index_order_abcd, ijkl) == lkji);
        }
      }
    }
  }
}

}  // namespace TestHelpers::TensorExpressions
