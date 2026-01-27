// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <utility>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComponentPlaceholder.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TestHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::tenex {

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 3 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti::a`, `ti::b`, and `ti::c`. The
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
/// If `ReturnLhsTensor == true`, the `tenex::evaluate` overload that returns
/// the LHS tensor will be tested. This, in turn, includes testing whether
/// `tenex::evaluate` is deducing the correct LHS tensor return type, where
/// `LhsSymmetry` is its expected symmetry and `LhsTensorIndexType` is its
/// expected list of indices.
///
/// If `ReturnLhsTensor == false`, the `tenex::evaluate` overload that takes a
/// LHS tensor as an argument will be tested. In this case, `LhsSymmetry` and
/// `LhsTensorIndexList` can be different from and will override what would be
/// automatically deduced from the RHS tensor expression. This is useful for
/// testing evaluations where the desired LHS tensor type would not
/// automatically be deduced from the RHS expression. For example, given some
/// tensor \f$L_{abc}\f$ with three spacetime indices, one can test whether
/// \f$L_{ijk} = ...\f$ correctly only assigns to the spatial-spatial-spatial
/// components of the tensor. Likewise, `ReturnLhsTensor == false` is
/// necessary to test cases where the LHS symmetry is different from what
/// would be deduced from the RHS expression.
///
/// \tparam ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not, which instead tests evaluation by
/// assigning to the result tensor passed in as an argument
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::B`
/// \tparam TensorIndexC the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::c`
/// \tparam LhsSymmetry the ::Symmetry of the LHS Tensor
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, auto& TensorIndexA,
          auto& TensorIndexB, auto& TensorIndexC,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_3_impl() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-5.0, 5.0);
  const size_t used_for_size = 3;
  using R_abc_type = Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList>;
  const auto R_abc = make_with_random_values<R_abc_type>(
      make_not_null(&generator), distribution, used_for_size);
  auto expected_L_abc =
      ReturnLhsTensor
          ? Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>{}
          : make_with_value<
                Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>>(
                used_for_size, component_placeholder_value<DataType>::value);

  const std::int32_t lhs_symmetry_element_a = tmpl::at_c<LhsSymmetry, 0>::value;
  const std::int32_t lhs_symmetry_element_b = tmpl::at_c<LhsSymmetry, 1>::value;
  const std::int32_t lhs_symmetry_element_c = tmpl::at_c<LhsSymmetry, 2>::value;
  using lhs_tensorindextype_a = tmpl::at_c<LhsTensorIndexTypeList, 0>;
  using lhs_tensorindextype_b = tmpl::at_c<LhsTensorIndexTypeList, 1>;
  using lhs_tensorindextype_c = tmpl::at_c<LhsTensorIndexTypeList, 2>;
  using rhs_tensorindextype_a = tmpl::at_c<RhsTensorIndexTypeList, 0>;
  using rhs_tensorindextype_b = tmpl::at_c<RhsTensorIndexTypeList, 1>;
  using rhs_tensorindextype_c = tmpl::at_c<RhsTensorIndexTypeList, 2>;

  std::array<std::pair<size_t, size_t>, 3> lhs_index_value_ranges{};
  lhs_index_value_ranges[0] =
      get_index_value_range<lhs_tensorindextype_a, TensorIndexA>();
  lhs_index_value_ranges[1] =
      get_index_value_range<lhs_tensorindextype_b, TensorIndexB>();
  lhs_index_value_ranges[2] =
      get_index_value_range<lhs_tensorindextype_c, TensorIndexC>();
  std::array<std::pair<size_t, size_t>, 3> rhs_index_value_ranges{};
  rhs_index_value_ranges[0] =
      get_index_value_range<rhs_tensorindextype_a, TensorIndexA>();
  rhs_index_value_ranges[1] =
      get_index_value_range<rhs_tensorindextype_b, TensorIndexB>();
  rhs_index_value_ranges[2] =
      get_index_value_range<rhs_tensorindextype_c, TensorIndexC>();

  for (size_t lhs_a = lhs_index_value_ranges[0].first,
              rhs_a = rhs_index_value_ranges[0].first;
       lhs_a <= lhs_index_value_ranges[0].second; lhs_a++, rhs_a++) {
    for (size_t lhs_b = lhs_index_value_ranges[1].first,
                rhs_b = rhs_index_value_ranges[1].first;
         lhs_b <= lhs_index_value_ranges[1].second; lhs_b++, rhs_b++) {
      for (size_t lhs_c = lhs_index_value_ranges[2].first,
                  rhs_c = rhs_index_value_ranges[2].first;
           lhs_c <= lhs_index_value_ranges[2].second; lhs_c++, rhs_c++) {
        expected_L_abc.get(lhs_a, lhs_b, lhs_c) =
            R_abc.get(rhs_a, rhs_b, rhs_c);
      }
    }
  }

  const auto rhs_expression = R_abc(TensorIndexA, TensorIndexB, TensorIndexC);

  // L_{abc} = R_{abc}
  using L_abc_type = Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>;
  L_abc_type L_abc(used_for_size);
  // component placeholder is used to detect which components have incorrectly
  // or correctly (in the case of using spatial or time indices for spacetime
  // indices) not been modified by evaluation of the RHS expression
  std::fill(L_abc.begin(), L_abc.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB, TensorIndexC>(
      make_not_null(&L_abc), rhs_expression);

  // L_{acb} = R_{abc}
  using L_acb_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_c,
               lhs_symmetry_element_b>;
  using L_acb_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_c,
                 lhs_tensorindextype_b>;
  using L_acb_type =
      Tensor<DataType, L_acb_symmetry, L_acb_tensorindextype_list>;
  L_acb_type L_acb(used_for_size);
  std::fill(L_acb.begin(), L_acb.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexC, TensorIndexB>(
      make_not_null(&L_acb), rhs_expression);

  // L_{bac} = R_{abc}
  using L_bac_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_a,
               lhs_symmetry_element_c>;
  using L_bac_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_a,
                 lhs_tensorindextype_c>;
  using L_bac_type =
      Tensor<DataType, L_bac_symmetry, L_bac_tensorindextype_list>;
  L_bac_type L_bac(used_for_size);
  std::fill(L_bac.begin(), L_bac.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA, TensorIndexC>(
      make_not_null(&L_bac), rhs_expression);

  // L_{bca} = R_{abc}
  using L_bca_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_c,
               lhs_symmetry_element_a>;
  using L_bca_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_c,
                 lhs_tensorindextype_a>;
  using L_bca_type =
      Tensor<DataType, L_bca_symmetry, L_bca_tensorindextype_list>;
  L_bca_type L_bca(used_for_size);
  std::fill(L_bca.begin(), L_bca.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexC, TensorIndexA>(
      make_not_null(&L_bca), rhs_expression);

  // L_{cab} = R_{abc}
  using L_cab_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_a,
               lhs_symmetry_element_b>;
  using L_cab_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_a,
                 lhs_tensorindextype_b>;
  using L_cab_type =
      Tensor<DataType, L_cab_symmetry, L_cab_tensorindextype_list>;
  L_cab_type L_cab(used_for_size);
  std::fill(L_cab.begin(), L_cab.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexA, TensorIndexB>(
      make_not_null(&L_cab), rhs_expression);

  // L_{cba} = R_{abc}
  using L_cba_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_b,
               lhs_symmetry_element_a>;
  using L_cba_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_b,
                 lhs_tensorindextype_a>;
  using L_cba_type =
      Tensor<DataType, L_cba_symmetry, L_cba_tensorindextype_list>;
  L_cba_type L_cba(used_for_size);
  std::fill(L_cba.begin(), L_cba.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexB, TensorIndexA>(
      make_not_null(&L_cba), rhs_expression);

  const size_t dim_a = tmpl::at_c<LhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<LhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<LhsTensorIndexTypeList, 2>::dim;

  // check LHS evaluated correctly
  for (size_t lhs_a = 0; lhs_a < dim_a; ++lhs_a) {
    for (size_t lhs_b = 0; lhs_b < dim_b; ++lhs_b) {
      for (size_t lhs_c = 0; lhs_c < dim_c; ++lhs_c) {
        const auto& expected_result = expected_L_abc.get(lhs_a, lhs_b, lhs_c);

        CHECK(L_abc.get(lhs_a, lhs_b, lhs_c) == expected_result);
        CHECK(L_acb.get(lhs_a, lhs_c, lhs_b) == expected_result);
        CHECK(L_bac.get(lhs_b, lhs_a, lhs_c) == expected_result);
        CHECK(L_bca.get(lhs_b, lhs_c, lhs_a) == expected_result);
        CHECK(L_cab.get(lhs_c, lhs_a, lhs_b) == expected_result);
        CHECK(L_cba.get(lhs_c, lhs_b, lhs_a) == expected_result);
      }
    }
  }

  // Test with Variables
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<
        ::Tags::TempTensor<0, R_abc_type>, ::Tags::TempTensor<1, L_abc_type>,
        ::Tags::TempTensor<2, L_acb_type>, ::Tags::TempTensor<3, L_bac_type>,
        ::Tags::TempTensor<4, L_bca_type>, ::Tags::TempTensor<5, L_cab_type>,
        ::Tags::TempTensor<6, L_cba_type>>>
        vars(used_for_size, std::numeric_limits<double>::signaling_NaN());

    R_abc_type& R_abc_temp = get<::Tags::TempTensor<0, R_abc_type>>(vars);
    R_abc_temp = R_abc;

    // L_{abc} = R_{abc}
    L_abc_type& L_abc_temp = get<::Tags::TempTensor<1, L_abc_type>>(vars);
    std::fill(L_abc_temp.begin(), L_abc_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB, TensorIndexC>(
        make_not_null(&L_abc_temp), rhs_expression);

    // L_{acb} = R_{abc}
    L_acb_type& L_acb_temp = get<::Tags::TempTensor<2, L_acb_type>>(vars);
    std::fill(L_acb_temp.begin(), L_acb_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexC, TensorIndexB>(
        make_not_null(&L_acb_temp), rhs_expression);

    // L_{bac} = R_{abc}
    L_bac_type& L_bac_temp = get<::Tags::TempTensor<3, L_bac_type>>(vars);
    std::fill(L_bac_temp.begin(), L_bac_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA, TensorIndexC>(
        make_not_null(&L_bac_temp), rhs_expression);

    // L_{bca} = R_{abc}
    L_bca_type& L_bca_temp = get<::Tags::TempTensor<4, L_bca_type>>(vars);
    std::fill(L_bca_temp.begin(), L_bca_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexC, TensorIndexA>(
        make_not_null(&L_bca_temp), rhs_expression);

    // L_{cab} = R_{abc}
    L_cab_type& L_cab_temp = get<::Tags::TempTensor<5, L_cab_type>>(vars);
    std::fill(L_cab_temp.begin(), L_cab_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexA, TensorIndexB>(
        make_not_null(&L_cab_temp), rhs_expression);

    // L_{cba} = R_{abc}
    L_cba_type& L_cba_temp = get<::Tags::TempTensor<6, L_cba_type>>(vars);
    std::fill(L_cba_temp.begin(), L_cba_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexB, TensorIndexA>(
        make_not_null(&L_cba_temp), rhs_expression);

    // check RHS wasn't modified
    CHECK(R_abc_temp == R_abc);

    // check LHS evaluated correctly
    for (size_t lhs_a = 0; lhs_a < dim_a; ++lhs_a) {
      for (size_t lhs_b = 0; lhs_b < dim_b; ++lhs_b) {
        for (size_t lhs_c = 0; lhs_c < dim_c; ++lhs_c) {
          const auto& expected_result = expected_L_abc.get(lhs_a, lhs_b, lhs_c);

          CHECK(L_abc_temp.get(lhs_a, lhs_b, lhs_c) == expected_result);
          CHECK(L_acb_temp.get(lhs_a, lhs_c, lhs_b) == expected_result);
          CHECK(L_bac_temp.get(lhs_b, lhs_a, lhs_c) == expected_result);
          CHECK(L_bca_temp.get(lhs_b, lhs_c, lhs_a) == expected_result);
          CHECK(L_cab_temp.get(lhs_c, lhs_a, lhs_b) == expected_result);
          CHECK(L_cba_temp.get(lhs_c, lhs_b, lhs_a) == expected_result);
        }
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluating single rank 3 Tensors on multiple
/// dimension combinations
///
/// We test various different symmetries across several functions to ensure that
/// the code works correctly with symmetries. This function tests one of the
/// following symmetries:
/// - <3, 2, 1>
/// - <2, 2, 1>
/// - <2, 1, 2>
/// - <2, 1, 1>
/// - <1, 1, 1>
///
/// \details See `test_evaluate_rank_3_impl` for general details.
///
/// \tparam ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not, which instead tests evaluation by
/// assigning to the result tensor passed in as an argument
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsIndexTypeList the RHS Tensor's integral list of `IndexType`s
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::B`
/// \tparam TensorIndexC the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::c`
/// \tparam Frame the frame of the tensor index
/// \tparam LhsSymmetry the ::Symmetry of the LHS Tensor
/// \tparam LhsIndexTypeList the LHS Tensor's integral list of `IndexType`s
template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsIndexTypeList, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, typename Frame,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsIndexTypeList = RhsIndexTypeList,
          Requires<std::is_same_v<RhsSymmetry, Symmetry<3, 2, 1>>> = nullptr>
void test_evaluate_rank_3() {
  constexpr IndexType rhs_indextype_a = tmpl::at_c<RhsIndexTypeList, 0>::value;
  constexpr IndexType rhs_indextype_b = tmpl::at_c<RhsIndexTypeList, 1>::value;
  constexpr IndexType rhs_indextype_c = tmpl::at_c<RhsIndexTypeList, 2>::value;
  constexpr IndexType lhs_indextype_a = tmpl::at_c<LhsIndexTypeList, 0>::value;
  constexpr IndexType lhs_indextype_b = tmpl::at_c<LhsIndexTypeList, 1>::value;
  constexpr IndexType lhs_indextype_c = tmpl::at_c<LhsIndexTypeList, 2>::value;

#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                               \
  test_evaluate_rank_3_impl<                                                  \
      ReturnLhsTensor, DataType, RhsSymmetry,                                 \
      index_list<                                                             \
                                                                              \
          ::Tensor_detail::TensorIndexType<DIM_A(data), TensorIndexA.valence, \
                                           Frame, rhs_indextype_a>,           \
          ::Tensor_detail::TensorIndexType<DIM_B(data), TensorIndexB.valence, \
                                           Frame, rhs_indextype_b>,           \
          ::Tensor_detail::TensorIndexType<DIM_C(data), TensorIndexC.valence, \
                                           Frame, rhs_indextype_c>>,          \
      TensorIndexA, TensorIndexB, TensorIndexC, LhsSymmetry,                  \
      index_list<                                                             \
          ::Tensor_detail::TensorIndexType<DIM_A(data), TensorIndexA.valence, \
                                           Frame, lhs_indextype_a>,           \
          ::Tensor_detail::TensorIndexType<DIM_B(data), TensorIndexB.valence, \
                                           Frame, lhs_indextype_b>,           \
          ::Tensor_detail::TensorIndexType<DIM_C(data), TensorIndexC.valence, \
                                           Frame, lhs_indextype_c>>>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3),
                          (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL

#undef DIM_C
#undef DIM_B
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsIndexTypeList, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, typename Frame,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsIndexTypeList = RhsIndexTypeList,
          Requires<std::is_same_v<RhsSymmetry, Symmetry<2, 2, 1>>> = nullptr>
void test_evaluate_rank_3() {
  constexpr IndexType rhs_indextype_a = tmpl::at_c<RhsIndexTypeList, 0>::value;
  constexpr IndexType rhs_indextype_b = tmpl::at_c<RhsIndexTypeList, 1>::value;
  constexpr IndexType rhs_indextype_c = tmpl::at_c<RhsIndexTypeList, 2>::value;
  constexpr IndexType lhs_indextype_a = tmpl::at_c<LhsIndexTypeList, 0>::value;
  constexpr IndexType lhs_indextype_b = tmpl::at_c<LhsIndexTypeList, 1>::value;
  constexpr IndexType lhs_indextype_c = tmpl::at_c<LhsIndexTypeList, 2>::value;

#define DIM_AB(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                                \
  test_evaluate_rank_3_impl<                                                   \
      ReturnLhsTensor, DataType, RhsSymmetry,                                  \
      index_list<                                                              \
          ::Tensor_detail::TensorIndexType<DIM_AB(data), TensorIndexA.valence, \
                                           Frame, rhs_indextype_a>,            \
          ::Tensor_detail::TensorIndexType<DIM_AB(data), TensorIndexB.valence, \
                                           Frame, rhs_indextype_b>,            \
          ::Tensor_detail::TensorIndexType<DIM_C(data), TensorIndexC.valence,  \
                                           Frame, rhs_indextype_c>>,           \
      TensorIndexA, TensorIndexB, TensorIndexC, LhsSymmetry,                   \
      index_list<                                                              \
          ::Tensor_detail::TensorIndexType<DIM_AB(data), TensorIndexA.valence, \
                                           Frame, lhs_indextype_a>,            \
          ::Tensor_detail::TensorIndexType<DIM_AB(data), TensorIndexB.valence, \
                                           Frame, lhs_indextype_b>,            \
          ::Tensor_detail::TensorIndexType<DIM_C(data), TensorIndexC.valence,  \
                                           Frame, lhs_indextype_c>>>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL

#undef DIM_C
#undef DIM_AB
}

/// \ingroup TestingFrameworkGroup
template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsIndexTypeList, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, typename Frame,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsIndexTypeList = RhsIndexTypeList,
          Requires<std::is_same_v<RhsSymmetry, Symmetry<1, 2, 1>>> = nullptr>
void test_evaluate_rank_3() {
  constexpr IndexType rhs_indextype_a = tmpl::at_c<RhsIndexTypeList, 0>::value;
  constexpr IndexType rhs_indextype_b = tmpl::at_c<RhsIndexTypeList, 1>::value;
  constexpr IndexType rhs_indextype_c = tmpl::at_c<RhsIndexTypeList, 2>::value;
  constexpr IndexType lhs_indextype_a = tmpl::at_c<LhsIndexTypeList, 0>::value;
  constexpr IndexType lhs_indextype_b = tmpl::at_c<LhsIndexTypeList, 1>::value;
  constexpr IndexType lhs_indextype_c = tmpl::at_c<LhsIndexTypeList, 2>::value;

#define DIM_AC(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                                \
  test_evaluate_rank_3_impl<                                                   \
      ReturnLhsTensor, DataType, RhsSymmetry,                                  \
      index_list<                                                              \
          ::Tensor_detail::TensorIndexType<DIM_AC(data), TensorIndexA.valence, \
                                           Frame, rhs_indextype_a>,            \
          ::Tensor_detail::TensorIndexType<DIM_B(data), TensorIndexB.valence,  \
                                           Frame, rhs_indextype_b>,            \
          ::Tensor_detail::TensorIndexType<DIM_AC(data), TensorIndexC.valence, \
                                           Frame, rhs_indextype_c>>,           \
      TensorIndexA, TensorIndexB, TensorIndexC, LhsSymmetry,                   \
      index_list<                                                              \
          ::Tensor_detail::TensorIndexType<DIM_AC(data), TensorIndexA.valence, \
                                           Frame, lhs_indextype_a>,            \
          ::Tensor_detail::TensorIndexType<DIM_B(data), TensorIndexB.valence,  \
                                           Frame, lhs_indextype_b>,            \
          ::Tensor_detail::TensorIndexType<DIM_AC(data), TensorIndexC.valence, \
                                           Frame, lhs_indextype_c>>>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL

#undef DIM_B
#undef DIM_AC
}

/// \ingroup TestingFrameworkGroup
template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsIndexTypeList, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, typename Frame,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsIndexTypeList = RhsIndexTypeList,
          Requires<std::is_same_v<RhsSymmetry, Symmetry<2, 1, 1>>> = nullptr>
void test_evaluate_rank_3() {
  constexpr IndexType rhs_indextype_a = tmpl::at_c<RhsIndexTypeList, 0>::value;
  constexpr IndexType rhs_indextype_b = tmpl::at_c<RhsIndexTypeList, 1>::value;
  constexpr IndexType rhs_indextype_c = tmpl::at_c<RhsIndexTypeList, 2>::value;
  constexpr IndexType lhs_indextype_a = tmpl::at_c<LhsIndexTypeList, 0>::value;
  constexpr IndexType lhs_indextype_b = tmpl::at_c<LhsIndexTypeList, 1>::value;
  constexpr IndexType lhs_indextype_c = tmpl::at_c<LhsIndexTypeList, 2>::value;

#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_BC(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                                \
  test_evaluate_rank_3_impl<                                                   \
      ReturnLhsTensor, DataType, RhsSymmetry,                                  \
      index_list<                                                              \
          ::Tensor_detail::TensorIndexType<DIM_A(data), TensorIndexA.valence,  \
                                           Frame, rhs_indextype_a>,            \
          ::Tensor_detail::TensorIndexType<DIM_BC(data), TensorIndexB.valence, \
                                           Frame, rhs_indextype_b>,            \
          ::Tensor_detail::TensorIndexType<DIM_BC(data), TensorIndexC.valence, \
                                           Frame, rhs_indextype_c>>,           \
      TensorIndexA, TensorIndexB, TensorIndexC, LhsSymmetry,                   \
      index_list<                                                              \
          ::Tensor_detail::TensorIndexType<DIM_A(data), TensorIndexA.valence,  \
                                           Frame, lhs_indextype_a>,            \
          ::Tensor_detail::TensorIndexType<DIM_BC(data), TensorIndexB.valence, \
                                           Frame, lhs_indextype_b>,            \
          ::Tensor_detail::TensorIndexType<DIM_BC(data), TensorIndexC.valence, \
                                           Frame, lhs_indextype_c>>>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL

#undef DIM_BC
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsIndexTypeList, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, typename Frame,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsIndexTypeList = RhsIndexTypeList,
          Requires<std::is_same_v<RhsSymmetry, Symmetry<1, 1, 1>>> = nullptr>
void test_evaluate_rank_3() {
  constexpr IndexType rhs_indextype_a = tmpl::at_c<RhsIndexTypeList, 0>::value;
  constexpr IndexType rhs_indextype_b = tmpl::at_c<RhsIndexTypeList, 1>::value;
  constexpr IndexType rhs_indextype_c = tmpl::at_c<RhsIndexTypeList, 2>::value;
  constexpr IndexType lhs_indextype_a = tmpl::at_c<LhsIndexTypeList, 0>::value;
  constexpr IndexType lhs_indextype_b = tmpl::at_c<LhsIndexTypeList, 1>::value;
  constexpr IndexType lhs_indextype_c = tmpl::at_c<LhsIndexTypeList, 2>::value;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                             \
  test_evaluate_rank_3_impl<                                                \
      ReturnLhsTensor, DataType, RhsSymmetry,                               \
      index_list<                                                           \
          ::Tensor_detail::TensorIndexType<DIM(data), TensorIndexA.valence, \
                                           Frame, rhs_indextype_a>,         \
          ::Tensor_detail::TensorIndexType<DIM(data), TensorIndexB.valence, \
                                           Frame, rhs_indextype_b>,         \
          ::Tensor_detail::TensorIndexType<DIM(data), TensorIndexC.valence, \
                                           Frame, rhs_indextype_c>>,        \
      TensorIndexA, TensorIndexB, TensorIndexC, LhsSymmetry,                \
      index_list<                                                           \
          ::Tensor_detail::TensorIndexType<DIM(data), TensorIndexA.valence, \
                                           Frame, lhs_indextype_a>,         \
          ::Tensor_detail::TensorIndexType<DIM(data), TensorIndexB.valence, \
                                           Frame, lhs_indextype_b>,         \
          ::Tensor_detail::TensorIndexType<DIM(data), TensorIndexC.valence, \
                                           Frame, lhs_indextype_c>>>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL

#undef DIM
}

}  // namespace TestHelpers::tenex
