// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
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
/// single rank 2 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details `TensorIndexA` and `TensorIndexB` can be any type of TensorIndex
/// and are not necessarily `ti_a` and `ti_b`. The "A" and "B" suffixes just
/// denote the ordering of the generic indices of the RHS tensor expression. In
/// the RHS tensor expression, it means `TensorIndexA` is the first index used
/// and `TensorIndexB` is the second index used.
///
/// If we consider the RHS tensor's generic indices to be (a, b), then this test
/// checks that the data in the evaluated LHS tensor is correct according to the
/// index orders of the LHS and RHS. The two possible cases that are checked are
/// when the LHS tensor is evaluated with index order (a, b) and when it is
/// evaluated with the index order (b, a).
///
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
template <typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, auto& TensorIndexA,
          auto& TensorIndexB>
void test_evaluate_rank_2_impl() noexcept {
  const size_t used_for_size = 5;
  Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList> R_ab(used_for_size);
  std::iota(R_ab.begin(), R_ab.end(), 0.0);

  // L_{ab} = R_{ab}
  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  using L_ab_type = decltype(R_ab);
  const L_ab_type L_ab_returned =
      ::TensorExpressions::evaluate<TensorIndexA, TensorIndexB>(
          R_ab(TensorIndexA, TensorIndexB));
  L_ab_type L_ab_filled{};
  ::TensorExpressions::evaluate<TensorIndexA, TensorIndexB>(
      make_not_null(&L_ab_filled), R_ab(TensorIndexA, TensorIndexB));

  // L_{ba} = R_{ab}
  using L_ba_tensorindextype_list =
      tmpl::list<tmpl::at_c<RhsTensorIndexTypeList, 1>,
                 tmpl::at_c<RhsTensorIndexTypeList, 0>>;
  using L_ba_type = Tensor<DataType, RhsSymmetry, L_ba_tensorindextype_list>;
  const L_ba_type L_ba_returned =
      ::TensorExpressions::evaluate<TensorIndexB, TensorIndexA>(
          R_ab(TensorIndexA, TensorIndexB));
  L_ba_type L_ba_filled{};
  ::TensorExpressions::evaluate<TensorIndexB, TensorIndexA>(
      make_not_null(&L_ba_filled), R_ab(TensorIndexA, TensorIndexB));

  const size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;

  for (size_t i = 0; i < dim_a; ++i) {
    for (size_t j = 0; j < dim_b; ++j) {
      // For L_{ab} = R_{ab}, check that L_{ij} == R_{ij}
      CHECK(L_ab_returned.get(i, j) == R_ab.get(i, j));
      CHECK(L_ab_filled.get(i, j) == R_ab.get(i, j));
      // For L_{ba} = R_{ab}, check that L_{ji} == R_{ij}
      CHECK(L_ba_returned.get(j, i) == R_ab.get(i, j));
      CHECK(L_ba_filled.get(j, i) == R_ab.get(i, j));
    }
  }

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    // L_{ab} = R_{ab}
    Variables<tmpl::list<::Tags::TempTensor<1, L_ab_type>>> L_ab_var{
        used_for_size};
    L_ab_type& L_ab_temp = get<::Tags::TempTensor<1, L_ab_type>>(L_ab_var);
    ::TensorExpressions::evaluate<TensorIndexA, TensorIndexB>(
        make_not_null(&L_ab_temp), R_ab(TensorIndexA, TensorIndexB));

    // L_{ba} = R_{ab}
    Variables<tmpl::list<::Tags::TempTensor<1, L_ba_type>>> L_ba_var{
        used_for_size};
    L_ba_type& L_ba_temp = get<::Tags::TempTensor<1, L_ba_type>>(L_ba_var);
    ::TensorExpressions::evaluate<TensorIndexB, TensorIndexA>(
        make_not_null(&L_ba_temp), R_ab(TensorIndexA, TensorIndexB));

    for (size_t i = 0; i < dim_a; ++i) {
      for (size_t j = 0; j < dim_b; ++j) {
        // For L_{ab} = R_{ab}, check that L_{ij} == R_{ij}
        CHECK(L_ab_temp.get(i, j) == R_ab.get(i, j));
        // For L_{ba} = R_{ab}, check that L_{ji} == R_{ij}
        CHECK(L_ba_temp.get(j, i) == R_ab.get(i, j));
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluating single rank 2 Tensors on multiple Frame
/// types and dimension combinations
///
/// We test nonsymmetric indices and symmetric indices across two functions to
/// ensure that the code works correctly with symmetries. This function tests
/// one of the following symmetries:
/// - <2, 1> (`test_evaluate_rank_2_no_symmetry`)
/// - <1, 1> (`test_evaluate_rank_2_symmetric`)
///
/// \details `TensorIndexA` and `TensorIndexB` can be any type of TensorIndex
/// and are not necessarily `ti_a` and `ti_b`. The "A" and "B" suffixes just
/// denote the ordering of the generic indices of the RHS tensor expression. In
/// the RHS tensor expression, it means `TensorIndexA` is the first index used
/// and `TensorIndexB` is the second index used.
///
/// Note: `test_evaluate_rank_2_symmetric` has fewer template parameters due to
/// the two indices having a shared \ref SpacetimeIndex "TensorIndexType" and
/// and valence
///
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam TensorIndexTypeA the \ref SpacetimeIndex "TensorIndexType" of the
/// first index of the RHS Tensor
/// \tparam TensorIndexTypeB the \ref SpacetimeIndex "TensorIndexType" of the
/// second index of the RHS Tensor
/// \tparam ValenceA the valence of the first index used on the RHS of the
/// TensorExpression
/// \tparam ValenceB the valence of the second index used on the RHS of the
/// TensorExpression
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexTypeA,
          template <size_t, UpLo, typename> class TensorIndexTypeB,
          UpLo ValenceA, UpLo ValenceB, auto& TensorIndexA, auto& TensorIndexB>
void test_evaluate_rank_2_no_symmetry() noexcept {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_2_IMPL(_, data)                         \
  test_evaluate_rank_2_impl<                                            \
      DataType, Symmetry<2, 1>,                                         \
      index_list<TensorIndexTypeA<DIM_A(data), ValenceA, FRAME(data)>,  \
                 TensorIndexTypeB<DIM_B(data), ValenceB, FRAME(data)>>, \
      TensorIndexA, TensorIndexB>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_2_IMPL, (1, 2, 3), (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_2_IMPL
#undef FRAME
#undef DIM_B
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_evaluate_rank_2_no_symmetry()
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexType, UpLo Valence,
          auto& TensorIndexA, auto& TensorIndexB>
void test_evaluate_rank_2_symmetric() noexcept {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_2_IMPL(_, data)                     \
  test_evaluate_rank_2_impl<                                        \
      DataType, Symmetry<1, 1>,                                     \
      index_list<TensorIndexType<DIM(data), Valence, FRAME(data)>,  \
                 TensorIndexType<DIM(data), Valence, FRAME(data)>>, \
      TensorIndexA, TensorIndexB>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_2_IMPL, (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_2_IMPL
#undef FRAME
#undef DIM
}

}  // namespace TestHelpers::TensorExpressions
