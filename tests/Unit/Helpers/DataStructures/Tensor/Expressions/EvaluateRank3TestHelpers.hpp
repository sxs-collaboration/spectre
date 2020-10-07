// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::TensorExpressions {

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 3 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti_a_t`, `ti_b_t`, and `ti_c_t`. The
/// "A", "B", and "C" suffixes just denote the ordering of the generic indices
/// of the RHS tensor expression. In the RHS tensor expression, it means
/// `TensorIndexA` is the first index used, `TensorIndexB` is the second index
/// used, and `TensorIndexC` is the third index used.
///
/// If we consider the RHS tensor's generic indices to be (a, b, c), then this
/// test checks that the data in the evaluated LHS tensor is correct according
/// to the index orders of the LHS and RHS. The possible cases that are checked
/// are when the LHS tensor is evaluated with index orders: (a, b, c),
/// (a, c, b), (b, a, c), (b, c, a), (c, a, b), and (c, b, a).
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
void test_evaluate_rank_3_impl(const TensorIndexA& tensorindex_a,
                               const TensorIndexB& tensorindex_b,
                               const TensorIndexC& tensorindex_c) noexcept {
  Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList> R_abc(5_st);
  std::iota(R_abc.begin(), R_abc.end(), 0.0);

  // Used for enforcing the ordering of the symmetry and TensorIndexTypes of the
  // LHS Tensor returned by `evaluate`
  const std::int32_t rhs_symmetry_element_a = tmpl::at_c<RhsSymmetry, 0>::value;
  const std::int32_t rhs_symmetry_element_b = tmpl::at_c<RhsSymmetry, 1>::value;
  const std::int32_t rhs_symmetry_element_c = tmpl::at_c<RhsSymmetry, 2>::value;
  using rhs_tensorindextype_a = tmpl::at_c<RhsTensorIndexTypeList, 0>;
  using rhs_tensorindextype_b = tmpl::at_c<RhsTensorIndexTypeList, 1>;
  using rhs_tensorindextype_c = tmpl::at_c<RhsTensorIndexTypeList, 2>;

  // L_{abc} = R_{abc}
  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  const Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList> L_abc =
      ::TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexC>(
          R_abc(tensorindex_a, tensorindex_b, tensorindex_c));

  // L_{acb} = R_{abc}
  using L_acb_symmetry =
      Symmetry<rhs_symmetry_element_a, rhs_symmetry_element_c,
               rhs_symmetry_element_b>;
  using L_acb_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_a, rhs_tensorindextype_c,
                 rhs_tensorindextype_b>;
  const Tensor<DataType, L_acb_symmetry, L_acb_tensorindextype_list> L_acb =
      ::TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexB>(
          R_abc(tensorindex_a, tensorindex_b, tensorindex_c));

  // L_{bac} = R_{abc}
  using L_bac_symmetry =
      Symmetry<rhs_symmetry_element_b, rhs_symmetry_element_a,
               rhs_symmetry_element_c>;
  using L_bac_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_b, rhs_tensorindextype_a,
                 rhs_tensorindextype_c>;
  const Tensor<DataType, L_bac_symmetry, L_bac_tensorindextype_list> L_bac =
      ::TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexC>(
          R_abc(tensorindex_a, tensorindex_b, tensorindex_c));

  // L_{bca} = R_{abc}
  using L_bca_symmetry =
      Symmetry<rhs_symmetry_element_b, rhs_symmetry_element_c,
               rhs_symmetry_element_a>;
  using L_bca_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_b, rhs_tensorindextype_c,
                 rhs_tensorindextype_a>;
  const Tensor<DataType, L_bca_symmetry, L_bca_tensorindextype_list> L_bca =
      ::TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexA>(
          R_abc(tensorindex_a, tensorindex_b, tensorindex_c));

  // L_{cab} = R_{abc}
  using L_cab_symmetry =
      Symmetry<rhs_symmetry_element_c, rhs_symmetry_element_a,
               rhs_symmetry_element_b>;
  using L_cab_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_c, rhs_tensorindextype_a,
                 rhs_tensorindextype_b>;
  const Tensor<DataType, L_cab_symmetry, L_cab_tensorindextype_list> L_cab =
      ::TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexB>(
          R_abc(tensorindex_a, tensorindex_b, tensorindex_c));

  // L_{cba} = R_{abc}
  using L_cba_symmetry =
      Symmetry<rhs_symmetry_element_c, rhs_symmetry_element_b,
               rhs_symmetry_element_a>;
  using L_cba_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_c, rhs_tensorindextype_b,
                 rhs_tensorindextype_a>;
  const Tensor<DataType, L_cba_symmetry, L_cba_tensorindextype_list> L_cba =
      ::TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexA>(
          R_abc(tensorindex_a, tensorindex_b, tensorindex_c));

  const size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<RhsTensorIndexTypeList, 2>::dim;

  for (size_t i = 0; i < dim_a; ++i) {
    for (size_t j = 0; j < dim_b; ++j) {
      for (size_t k = 0; k < dim_c; ++k) {
        // For L_{abc} = R_{abc}, check that L_{ijk} == R_{ijk}
        CHECK(L_abc.get(i, j, k) == R_abc.get(i, j, k));
        // For L_{acb} = R_{abc}, check that L_{ikj} == R_{ijk}
        CHECK(L_acb.get(i, k, j) == R_abc.get(i, j, k));
        // For L_{bac} = R_{abc}, check that L_{jik} == R_{ijk}
        CHECK(L_bac.get(j, i, k) == R_abc.get(i, j, k));
        // For L_{bca} = R_{abc}, check that L_{jki} == R_{ijk}
        CHECK(L_bca.get(j, k, i) == R_abc.get(i, j, k));
        // For L_{cab} = R_{abc}, check that L_{kij} == R_{ijk}
        CHECK(L_cab.get(k, i, j) == R_abc.get(i, j, k));
        // For L_{cba} = R_{abc}, check that L_{kji} == R_{ijk}
        CHECK(L_cba.get(k, j, i) == R_abc.get(i, j, k));
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluating single rank 3 Tensors on multiple Frame
/// types and dimension combinations
///
/// We test various different symmetries across several functions to ensure that
/// the code works correctly with symmetries. This function tests one of the
/// following symmetries:
/// - <3, 2, 1> (`test_evaluate_rank_3_no_symmetry`)
/// - <2, 2, 1> (`test_evaluate_rank_3_ab_symmetry`)
/// - <2, 1, 2> (`test_evaluate_rank_3_ac_symmetry`)
/// - <2, 1, 1> (`test_evaluate_rank_3_bc_symmetry`)
/// - <1, 1, 1> (`test_evaluate_rank_3_abc_symmetry`)
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti_a_t`, `ti_b_t`, and `ti_c_t`. The
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
void test_evaluate_rank_3_no_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(2, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(3, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                          \
  test_evaluate_rank_3_impl<                                             \
      DataType, Symmetry<3, 2, 1>,                                       \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,   \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,   \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3),
                          (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef FRAME
#undef DIM_C
#undef DIM_B
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_evaluate_rank_3_no_symmetry()
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexTypeAB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceAB, UpLo ValenceC, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_evaluate_rank_3_ab_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
#define DIM_AB(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                            \
  test_evaluate_rank_3_impl<                                               \
      DataType, Symmetry<2, 2, 1>,                                         \
      index_list<TensorIndexTypeAB<DIM_AB(data), ValenceAB, FRAME(data)>,  \
                 TensorIndexTypeAB<DIM_AB(data), ValenceAB, FRAME(data)>,  \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>>(   \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef FRAME
#undef DIM_C
#undef DIM_AB
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_evaluate_rank_3_no_symmetry()
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexTypeAC,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          UpLo ValenceAC, UpLo ValenceB, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_evaluate_rank_3_ac_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
#define DIM_AC(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                             \
  test_evaluate_rank_3_impl<                                                \
      DataType, Symmetry<2, 1, 2>,                                          \
      index_list<TensorIndexTypeAC<DIM_AC(data), ValenceAC, FRAME(data)>,   \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,      \
                 TensorIndexTypeAC<DIM_AC(data), ValenceAC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef FRAME
#undef DIM_B
#undef DIM_AC
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_evaluate_rank_3_no_symmetry()
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeBC,
          UpLo ValenceA, UpLo ValenceBC, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC>
void test_evaluate_rank_3_bc_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_BC(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                             \
  test_evaluate_rank_3_impl<                                                \
      DataType, Symmetry<2, 1, 1>,                                          \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,      \
                 TensorIndexTypeBC<DIM_BC(data), ValenceBC, FRAME(data)>,   \
                 TensorIndexTypeBC<DIM_BC(data), ValenceBC, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef FRAME
#undef DIM_BC
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_evaluate_rank_3_no_symmetry()
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexType, UpLo Valence,
          typename TensorIndexA, typename TensorIndexB, typename TensorIndexC>
void test_evaluate_rank_3_abc_symmetry(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c) noexcept {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                      \
  test_evaluate_rank_3_impl<                                         \
      DataType, Symmetry<1, 1, 1>,                                   \
      index_list<TensorIndexType<DIM(data), Valence, FRAME(data)>,   \
                 TensorIndexType<DIM(data), Valence, FRAME(data)>,   \
                 TensorIndexType<DIM(data), Valence, FRAME(data)>>>( \
      tensorindex_a, tensorindex_b, tensorindex_c);

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef FRAME
#undef DIM
}

}  // namespace TestHelpers::TensorExpressions
