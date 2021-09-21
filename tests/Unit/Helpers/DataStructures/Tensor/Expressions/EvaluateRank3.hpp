// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::TensorExpressions {

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 3 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti_a`, `ti_b`, and `ti_c`. The
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
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \tparam TensorIndexC the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
template <typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, auto& TensorIndexA,
          auto& TensorIndexB, auto& TensorIndexC>
void test_evaluate_rank_3_impl() noexcept {
  const size_t used_for_size = 5;
  Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList> R_abc(used_for_size);
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
  using L_abc_type = Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList>;
  const L_abc_type L_abc_returned =
      ::TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexC>(
          R_abc(TensorIndexA, TensorIndexB, TensorIndexC));
  L_abc_type L_abc_filled{};
  ::TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexC>(
      make_not_null(&L_abc_filled),
      R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

  // L_{acb} = R_{abc}
  using L_acb_symmetry =
      Symmetry<rhs_symmetry_element_a, rhs_symmetry_element_c,
               rhs_symmetry_element_b>;
  using L_acb_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_a, rhs_tensorindextype_c,
                 rhs_tensorindextype_b>;
  using L_acb_type =
      Tensor<DataType, L_acb_symmetry, L_acb_tensorindextype_list>;
  const L_acb_type L_acb_returned =
      ::TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexB>(
          R_abc(TensorIndexA, TensorIndexB, TensorIndexC));
  L_acb_type L_acb_filled{};
  ::TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexB>(
      make_not_null(&L_acb_filled),
      R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

  // L_{bac} = R_{abc}
  using L_bac_symmetry =
      Symmetry<rhs_symmetry_element_b, rhs_symmetry_element_a,
               rhs_symmetry_element_c>;
  using L_bac_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_b, rhs_tensorindextype_a,
                 rhs_tensorindextype_c>;
  using L_bac_type =
      Tensor<DataType, L_bac_symmetry, L_bac_tensorindextype_list>;
  const L_bac_type L_bac_returned =
      ::TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexC>(
          R_abc(TensorIndexA, TensorIndexB, TensorIndexC));
  L_bac_type L_bac_filled{};
  ::TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexC>(
      make_not_null(&L_bac_filled),
      R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

  // L_{bca} = R_{abc}
  using L_bca_symmetry =
      Symmetry<rhs_symmetry_element_b, rhs_symmetry_element_c,
               rhs_symmetry_element_a>;
  using L_bca_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_b, rhs_tensorindextype_c,
                 rhs_tensorindextype_a>;
  using L_bca_type =
      Tensor<DataType, L_bca_symmetry, L_bca_tensorindextype_list>;
  const L_bca_type L_bca_returned =
      ::TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexA>(
          R_abc(TensorIndexA, TensorIndexB, TensorIndexC));
  L_bca_type L_bca_filled{};
  ::TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexA>(
      make_not_null(&L_bca_filled),
      R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

  // L_{cab} = R_{abc}
  using L_cab_symmetry =
      Symmetry<rhs_symmetry_element_c, rhs_symmetry_element_a,
               rhs_symmetry_element_b>;
  using L_cab_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_c, rhs_tensorindextype_a,
                 rhs_tensorindextype_b>;
  using L_cab_type =
      Tensor<DataType, L_cab_symmetry, L_cab_tensorindextype_list>;
  const L_cab_type L_cab_returned =
      ::TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexB>(
          R_abc(TensorIndexA, TensorIndexB, TensorIndexC));
  L_cab_type L_cab_filled{};
  ::TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexB>(
      make_not_null(&L_cab_filled),
      R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

  // L_{cba} = R_{abc}
  using L_cba_symmetry =
      Symmetry<rhs_symmetry_element_c, rhs_symmetry_element_b,
               rhs_symmetry_element_a>;
  using L_cba_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_c, rhs_tensorindextype_b,
                 rhs_tensorindextype_a>;
  using L_cba_type =
      Tensor<DataType, L_cba_symmetry, L_cba_tensorindextype_list>;
  const L_cba_type L_cba_returned =
      ::TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexA>(
          R_abc(TensorIndexA, TensorIndexB, TensorIndexC));
  L_cba_type L_cba_filled{};
  ::TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexA>(
      make_not_null(&L_cba_filled),
      R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

  const size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<RhsTensorIndexTypeList, 2>::dim;

  for (size_t i = 0; i < dim_a; ++i) {
    for (size_t j = 0; j < dim_b; ++j) {
      for (size_t k = 0; k < dim_c; ++k) {
        // For L_{abc} = R_{abc}, check that L_{ijk} == R_{ijk}
        CHECK(L_abc_returned.get(i, j, k) == R_abc.get(i, j, k));
        CHECK(L_abc_filled.get(i, j, k) == R_abc.get(i, j, k));
        // For L_{acb} = R_{abc}, check that L_{ikj} == R_{ijk}
        CHECK(L_acb_returned.get(i, k, j) == R_abc.get(i, j, k));
        CHECK(L_acb_filled.get(i, k, j) == R_abc.get(i, j, k));
        // For L_{bac} = R_{abc}, check that L_{jik} == R_{ijk}
        CHECK(L_bac_returned.get(j, i, k) == R_abc.get(i, j, k));
        CHECK(L_bac_filled.get(j, i, k) == R_abc.get(i, j, k));
        // For L_{bca} = R_{abc}, check that L_{jki} == R_{ijk}
        CHECK(L_bca_returned.get(j, k, i) == R_abc.get(i, j, k));
        CHECK(L_bca_filled.get(j, k, i) == R_abc.get(i, j, k));
        // For L_{cab} = R_{abc}, check that L_{kij} == R_{ijk}
        CHECK(L_cab_returned.get(k, i, j) == R_abc.get(i, j, k));
        CHECK(L_cab_filled.get(k, i, j) == R_abc.get(i, j, k));
        // For L_{cba} = R_{abc}, check that L_{kji} == R_{ijk}
        CHECK(L_cba_returned.get(k, j, i) == R_abc.get(i, j, k));
        CHECK(L_cba_filled.get(k, j, i) == R_abc.get(i, j, k));
      }
    }
  }

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    // L_{abc} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_abc_type>>> L_abc_var{
        used_for_size};
    L_abc_type& L_abc_temp = get<::Tags::TempTensor<1, L_abc_type>>(L_abc_var);
    ::TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexC>(
        make_not_null(&L_abc_temp),
        R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

    // L_{acb} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_acb_type>>> L_acb_var{
        used_for_size};
    L_acb_type& L_acb_temp = get<::Tags::TempTensor<1, L_acb_type>>(L_acb_var);
    ::TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexB>(
        make_not_null(&L_acb_temp),
        R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

    // L_{bac} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_bac_type>>> L_bac_var{
        used_for_size};
    L_bac_type& L_bac_temp = get<::Tags::TempTensor<1, L_bac_type>>(L_bac_var);
    ::TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexC>(
        make_not_null(&L_bac_temp),
        R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

    // L_{bca} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_bca_type>>> L_bca_var{
        used_for_size};
    L_bca_type& L_bca_temp = get<::Tags::TempTensor<1, L_bca_type>>(L_bca_var);
    ::TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexA>(
        make_not_null(&L_bca_temp),
        R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

    // L_{cab} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_cab_type>>> L_cab_var{
        used_for_size};
    L_cab_type& L_cab_temp = get<::Tags::TempTensor<1, L_cab_type>>(L_cab_var);
    ::TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexB>(
        make_not_null(&L_cab_temp),
        R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

    // L_{cba} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_cba_type>>> L_cba_var{
        used_for_size};
    L_cba_type& L_cba_temp = get<::Tags::TempTensor<1, L_cba_type>>(L_cba_var);
    ::TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexA>(
        make_not_null(&L_cba_temp),
        R_abc(TensorIndexA, TensorIndexB, TensorIndexC));

    for (size_t i = 0; i < dim_a; ++i) {
      for (size_t j = 0; j < dim_b; ++j) {
        for (size_t k = 0; k < dim_c; ++k) {
          // For L_{abc} = R_{abc}, check that L_{ijk} == R_{ijk}
          CHECK(L_abc_temp.get(i, j, k) == R_abc.get(i, j, k));
          // For L_{acb} = R_{abc}, check that L_{ikj} == R_{ijk}
          CHECK(L_acb_temp.get(i, k, j) == R_abc.get(i, j, k));
          // For L_{bac} = R_{abc}, check that L_{jik} == R_{ijk}
          CHECK(L_bac_temp.get(j, i, k) == R_abc.get(i, j, k));
          // For L_{bca} = R_{abc}, check that L_{jki} == R_{ijk}
          CHECK(L_bca_temp.get(j, k, i) == R_abc.get(i, j, k));
          // For L_{cab} = R_{abc}, check that L_{kij} == R_{ijk}
          CHECK(L_cab_temp.get(k, i, j) == R_abc.get(i, j, k));
          // For L_{cba} = R_{abc}, check that L_{kji} == R_{ijk}
          CHECK(L_cba_temp.get(k, j, i) == R_abc.get(i, j, k));
        }
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
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \tparam TensorIndexC the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          template <size_t, UpLo, typename> class TensorIndexTypeC,
          UpLo ValenceA, UpLo ValenceB, UpLo ValenceC, auto& TensorIndexA,
          auto& TensorIndexB, auto& TensorIndexC>
void test_evaluate_rank_3_no_symmetry() noexcept {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(2, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(3, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                         \
  test_evaluate_rank_3_impl<                                            \
      DataType, Symmetry<3, 2, 1>,                                      \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,  \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,  \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>, \
      TensorIndexA, TensorIndexB, TensorIndexC>();

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
          UpLo ValenceAB, UpLo ValenceC, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC>
void test_evaluate_rank_3_ab_symmetry() noexcept {
#define DIM_AB(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                           \
  test_evaluate_rank_3_impl<                                              \
      DataType, Symmetry<2, 2, 1>,                                        \
      index_list<TensorIndexTypeAB<DIM_AB(data), ValenceAB, FRAME(data)>, \
                 TensorIndexTypeAB<DIM_AB(data), ValenceAB, FRAME(data)>, \
                 TensorIndexTypeC<DIM_C(data), ValenceC, FRAME(data)>>,   \
      TensorIndexA, TensorIndexB, TensorIndexC>();

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
          UpLo ValenceAC, UpLo ValenceB, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC>
void test_evaluate_rank_3_ac_symmetry() noexcept {
#define DIM_AC(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                            \
  test_evaluate_rank_3_impl<                                               \
      DataType, Symmetry<2, 1, 2>,                                         \
      index_list<TensorIndexTypeAC<DIM_AC(data), ValenceAC, FRAME(data)>,  \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>,     \
                 TensorIndexTypeAC<DIM_AC(data), ValenceAC, FRAME(data)>>, \
      TensorIndexA, TensorIndexB, TensorIndexC>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef FRAME
#undef DIM_B
#undef DIM_AC
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_evaluate_rank_3_no_symmetry()
template <
    typename DataType, template <size_t, UpLo, typename> class TensorIndexTypeA,
    template <size_t, UpLo, typename> class TensorIndexTypeBC, UpLo ValenceA,
    UpLo ValenceBC, auto& TensorIndexA, auto& TensorIndexB, auto& TensorIndexC>
void test_evaluate_rank_3_bc_symmetry() noexcept {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_BC(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                            \
  test_evaluate_rank_3_impl<                                               \
      DataType, Symmetry<2, 1, 1>,                                         \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,     \
                 TensorIndexTypeBC<DIM_BC(data), ValenceBC, FRAME(data)>,  \
                 TensorIndexTypeBC<DIM_BC(data), ValenceBC, FRAME(data)>>, \
      TensorIndexA, TensorIndexB, TensorIndexC>();

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
          auto& TensorIndexA, auto& TensorIndexB, auto& TensorIndexC>
void test_evaluate_rank_3_abc_symmetry() noexcept {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                     \
  test_evaluate_rank_3_impl<                                        \
      DataType, Symmetry<1, 1, 1>,                                  \
      index_list<TensorIndexType<DIM(data), Valence, FRAME(data)>,  \
                 TensorIndexType<DIM(data), Valence, FRAME(data)>,  \
                 TensorIndexType<DIM(data), Valence, FRAME(data)>>, \
      TensorIndexA, TensorIndexB, TensorIndexC>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef FRAME
#undef DIM
}

}  // namespace TestHelpers::TensorExpressions
