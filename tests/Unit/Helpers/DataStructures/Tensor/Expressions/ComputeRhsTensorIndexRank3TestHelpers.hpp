// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::TensorExpressions {

/// \ingroup TestingFrameworkGroup
/// \brief Test that the computed tensor multi-index of a rank 3 RHS Tensor is
/// equivalent to the given LHS tensor multi-index, according to the order of
/// their generic indices
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti_a`, `ti_b`, and `ti_c`. The
/// "A", "B", and "C" suffixes just denote the ordering of the generic indices
/// of the RHS tensor expression. In the RHS tensor expression, it means
/// `TensorIndexA` is the first index used, `TensorIndexB` is the second index
/// used, and `TensorIndexC` is the third index used.
///
/// If we consider the RHS tensor's generic indices to be (a, b, c), the
/// possible orderings of the LHS tensor's generic indices are: (a, b, c),
/// (a, c, b), (b, a, c), (b, c, a), (c, a, b), and (c, b, a). For each of these
/// cases, this test checks that for each LHS component's tensor multi-index,
/// the equivalent RHS tensor multi-index is correctly computed.
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
template <typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_impl(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
  const Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList> rhs_tensor(5_st);
  // Get TensorExpression from RHS tensor
  const auto R_abc = rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c);

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

  const size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<RhsTensorIndexTypeList, 2>::dim;

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
        CHECK(R_abc.compute_rhs_tensor_index(index_order_abc, index_order_abc,
                                             ijk) == ijk);
        // For L_{acb} = R_{abc}, check that L_{ijk} == R_{ikj}
        CHECK(R_abc.compute_rhs_tensor_index(index_order_acb, index_order_abc,
                                             ijk) == ikj);
        // For L_{bac} = R_{abc}, check that L_{ijk} == R_{jik}
        CHECK(R_abc.compute_rhs_tensor_index(index_order_bac, index_order_abc,
                                             ijk) == jik);
        // For L_{bca} = R_{abc}, check that L_{ijk} == R_{kij}
        CHECK(R_abc.compute_rhs_tensor_index(index_order_bca, index_order_abc,
                                             ijk) == kij);
        // For L_{cab} = R_{abc}, check that L_{ijk} == R_{jki}
        CHECK(R_abc.compute_rhs_tensor_index(index_order_cab, index_order_abc,
                                             ijk) == jki);
        // For L_{cba} = R_{abc}, check that L_{ijk} == R_{kji}
        CHECK(R_abc.compute_rhs_tensor_index(index_order_cba, index_order_abc,
                                             ijk) == kji);
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of computing the RHS tensor multi-index equivalent of
/// the LHS tensor multi-index with rank 3 Tensors on multiple Frame types and
/// dimension combinations for nonsymmetric indices
///
/// We test various different symmetries across several functions to ensure that
/// the code works correctly with symmetries. This function tests one of the
/// following symmetries:
/// - <3, 2, 1> (`test_compute_rhs_tensor_index_rank_3_no_symmetry`)
/// - <2, 2, 1> (`test_compute_rhs_tensor_index_rank_3_ab_symmetry`)
/// - <2, 1, 2> (`test_compute_rhs_tensor_index_rank_3_ac_symmetry`)
/// - <2, 1, 1> (`test_compute_rhs_tensor_index_rank_3_bc_symmetry`)
/// - <1, 1, 1> (`test_compute_rhs_tensor_index_rank_3_abc_symmetry`)
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti_a`, `ti_b`, and `ti_c`. The
/// "A", "B", and "C" suffixes just denote the ordering of the generic indices
/// of the RHS tensor expression. In the RHS tensor expression, it means
/// `TensorIndexA` is the first index used, `TensorIndexB` is the second index
/// used, and `TensorIndexC` is the third index used.
///
/// Note: the functions dealing with symmetric indices have fewer template
/// parameters due to the indices having a shared \ref SpacetimeIndex
/// "TensorIndexType" and valence
///
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam TensorIndexTypeA the \ref SpacetimeIndex "TensorIndexType" of the
/// first index of the RHS Tensor
/// \tparam TensorIndexTypeB the \ref SpacetimeIndex "TensorIndexType" of the
/// second index of the RHS Tensor
/// \tparam TensorIndexTypeC the \ref SpacetimeIndex "TensorIndexType" of the
/// third index of the RHS Tensor
/// \tparam ValenceA the valence of the first index used on the RHS of the
/// TensorExpression
/// \tparam ValenceB the valence of the second index used on the RHS of the
/// TensorExpression
/// \tparam ValenceC the valence of the third index used on the RHS of the
/// TensorExpression
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \param tensorindex_c the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_no_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(2, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(3, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL(_, data)          \
  test_compute_rhs_tensor_index_rank_3_impl<                             \
      DataType, Symmetry<3, 2, 1>,                                       \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,   \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,   \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL,
                          (1, 2, 3), (1, 2, 3), (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL
#undef FRAME
#undef DIM_C
#undef DIM_B
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_compute_rhs_tensor_index_rank_3_no_symmetry()
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexTypeAB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceAB, UpLo ValenceC, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_ab_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
#define DIM_AB(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL(_, data)            \
  test_compute_rhs_tensor_index_rank_3_impl<                               \
      DataType, Symmetry<2, 2, 1>,                                         \
      index_list<TensorIndexTypeAB<DIM_AB(data), ValenceAB, FRAME(data)>,  \
                 TensorIndexTypeAB<DIM_AB(data), ValenceAB, FRAME(data)>,  \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>>(   \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL,
                          (1, 2, 3), (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL
#undef FRAME
#undef DIM_C
#undef DIM_AB
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_compute_rhs_tensor_index_rank_3_no_symmetry()
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexTypeAC,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          UpLo ValenceAC, UpLo ValenceB, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_ac_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
#define DIM_AC(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL(_, data)             \
  test_compute_rhs_tensor_index_rank_3_impl<                                \
      DataType, Symmetry<2, 1, 2>,                                          \
      index_list<TensorIndexTypeAC<DIM_AC(data), ValenceAC, FRAME(data)>,   \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,      \
                 TensorIndexTypeAC<DIM_AC(data), ValenceAC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL,
                          (1, 2, 3), (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL
#undef FRAME
#undef DIM_B
#undef DIM_AC
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_compute_rhs_tensor_index_rank_3_no_symmetry()
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeBC,
          UpLo ValenceA, UpLo ValenceBC, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_bc_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_BC(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL(_, data)             \
  test_compute_rhs_tensor_index_rank_3_impl<                                \
      DataType, Symmetry<2, 1, 1>,                                          \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,      \
                 TensorIndexTypeBC<DIM_BC(data), ValenceBC, FRAME(data)>,   \
                 TensorIndexTypeBC<DIM_BC(data), ValenceBC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL,
                          (1, 2, 3), (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL
#undef FRAME
#undef DIM_BC
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_compute_rhs_tensor_index_rank_3_no_symmetry()
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexType, UpLo Valence,
          typename TensorIndexA, typename TensorIndexB, typename TensorIndexC>
void test_compute_rhs_tensor_index_rank_3_abc_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL(_, data)      \
  test_compute_rhs_tensor_index_rank_3_impl<                         \
      DataType, Symmetry<1, 1, 1>,                                   \
      index_list<TensorIndexType<DIM(data), Valence, FRAME(data)>,   \
                 TensorIndexType<DIM(data), Valence, FRAME(data)>,   \
                 TensorIndexType<DIM(data), Valence, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL,
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_COMPUTE_RHS_TENSOR_INDEX_RANK_3_IMPL
#undef FRAME
#undef DIM
}

}  // namespace TestHelpers::TensorExpressions
