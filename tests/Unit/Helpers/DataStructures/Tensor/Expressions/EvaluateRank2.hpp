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
/// single rank 2 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details See `test_evaluate_rank_3_impl` for general details.
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
/// \tparam LhsSymmetry the ::Symmetry of the LHS Tensor
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, auto& TensorIndexA,
          auto& TensorIndexB, typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_2_impl() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-5.0, 5.0);
  const size_t used_for_size = 3;
  using R_ab_type = Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList>;
  const auto R_ab = make_with_random_values<R_ab_type>(
      make_not_null(&generator), distribution, used_for_size);
  auto expected_L_ab =
      ReturnLhsTensor
          ? Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>{}
          : make_with_value<
                Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>>(
                used_for_size, component_placeholder_value<DataType>::value);

  const std::int32_t lhs_symmetry_element_a = tmpl::at_c<LhsSymmetry, 0>::value;
  const std::int32_t lhs_symmetry_element_b = tmpl::at_c<LhsSymmetry, 1>::value;
  using lhs_tensorindextype_a = tmpl::at_c<LhsTensorIndexTypeList, 0>;
  using lhs_tensorindextype_b = tmpl::at_c<LhsTensorIndexTypeList, 1>;
  using rhs_tensorindextype_a = tmpl::at_c<RhsTensorIndexTypeList, 0>;
  using rhs_tensorindextype_b = tmpl::at_c<RhsTensorIndexTypeList, 1>;

  std::array<std::pair<size_t, size_t>, 2> lhs_index_value_ranges{};
  lhs_index_value_ranges[0] =
      get_index_value_range<lhs_tensorindextype_a, TensorIndexA>();
  lhs_index_value_ranges[1] =
      get_index_value_range<lhs_tensorindextype_b, TensorIndexB>();
  std::array<std::pair<size_t, size_t>, 2> rhs_index_value_ranges{};
  rhs_index_value_ranges[0] =
      get_index_value_range<rhs_tensorindextype_a, TensorIndexA>();
  rhs_index_value_ranges[1] =
      get_index_value_range<rhs_tensorindextype_b, TensorIndexB>();

  for (size_t lhs_a = lhs_index_value_ranges[0].first,
              rhs_a = rhs_index_value_ranges[0].first;
       lhs_a <= lhs_index_value_ranges[0].second; lhs_a++, rhs_a++) {
    for (size_t lhs_b = lhs_index_value_ranges[1].first,
                rhs_b = rhs_index_value_ranges[1].first;
         lhs_b <= lhs_index_value_ranges[1].second; lhs_b++, rhs_b++) {
      expected_L_ab.get(lhs_a, lhs_b) = R_ab.get(rhs_a, rhs_b);
    }
  }

  const auto rhs_expression = R_ab(TensorIndexA, TensorIndexB);

  // L_{ab} = R_{ab}
  using L_ab_type = Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>;
  L_ab_type L_ab(used_for_size);
  // component placeholder is used to detect which components have incorrectly
  // or correctly (in the case of using spatial or time indices for spacetime
  // indices) not been modified by evaluation of the RHS expression
  std::fill(L_ab.begin(), L_ab.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB>(
      make_not_null(&L_ab), rhs_expression);

  // L_{ba} = R_{ab}
  using L_ba_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_a>;
  using L_ba_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_a>;
  using L_ba_type = Tensor<DataType, L_ba_symmetry, L_ba_tensorindextype_list>;
  L_ba_type L_ba(used_for_size);
  std::fill(L_ba.begin(), L_ba.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA>(
      make_not_null(&L_ba), rhs_expression);

  const size_t dim_a = tmpl::at_c<LhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<LhsTensorIndexTypeList, 1>::dim;

  // check LHS evaluated correctly
  for (size_t lhs_a = 0; lhs_a < dim_a; ++lhs_a) {
    for (size_t lhs_b = 0; lhs_b < dim_b; ++lhs_b) {
      const auto& expected_result = expected_L_ab.get(lhs_a, lhs_b);

      CHECK(L_ab.get(lhs_a, lhs_b) == expected_result);
      CHECK(L_ba.get(lhs_b, lhs_a) == expected_result);
    }
  }

  // Test with Variables
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<::Tags::TempTensor<0, R_ab_type>,
                         ::Tags::TempTensor<1, L_ab_type>,
                         ::Tags::TempTensor<2, L_ba_type>>>
        vars(used_for_size, std::numeric_limits<double>::signaling_NaN());

    R_ab_type& R_ab_temp = get<::Tags::TempTensor<0, R_ab_type>>(vars);
    R_ab_temp = R_ab;

    // L_{ab} = R_{ab}
    L_ab_type& L_ab_temp = get<::Tags::TempTensor<1, L_ab_type>>(vars);
    std::fill(L_ab_temp.begin(), L_ab_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB>(
        make_not_null(&L_ab_temp), rhs_expression);

    // L_{ba} = R_{ab}
    L_ba_type& L_ba_temp = get<::Tags::TempTensor<2, L_ba_type>>(vars);
    std::fill(L_ba_temp.begin(), L_ba_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA>(
        make_not_null(&L_ba_temp), rhs_expression);

    // check RHS wasn't modified
    CHECK(R_ab_temp == R_ab);

    // check LHS evaluated correctly
    for (size_t lhs_a = 0; lhs_a < dim_a; ++lhs_a) {
      for (size_t lhs_b = 0; lhs_b < dim_b; ++lhs_b) {
        const auto& expected_result = expected_L_ab.get(lhs_a, lhs_b);

        CHECK(L_ab_temp.get(lhs_a, lhs_b) == expected_result);
        CHECK(L_ba_temp.get(lhs_b, lhs_a) == expected_result);
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluating single rank 2 Tensors on multiple
/// dimension combinations
///
/// We test nonsymmetric indices and symmetric indices across two functions to
/// ensure that the code works correctly with symmetries. This function tests
/// one of the following symmetries:
/// - <2, 1>
/// - <1, 1>
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
/// \tparam Frame the frame of the tensor index
/// \tparam LhsSymmetry the ::Symmetry of the LHS Tensor
/// \tparam LhsIndexTypeList the LHS Tensor's integral list of `IndexType`s
template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsIndexTypeList, auto& TensorIndexA, auto& TensorIndexB,
          typename Frame, typename LhsSymmetry = RhsSymmetry,
          typename LhsIndexTypeList = RhsIndexTypeList,
          Requires<std::is_same_v<RhsSymmetry, Symmetry<2, 1>>> = nullptr>
void test_evaluate_rank_2() {
  constexpr IndexType rhs_indextype_a = tmpl::at_c<RhsIndexTypeList, 0>::value;
  constexpr IndexType rhs_indextype_b = tmpl::at_c<RhsIndexTypeList, 1>::value;
  constexpr IndexType lhs_indextype_a = tmpl::at_c<LhsIndexTypeList, 0>::value;
  constexpr IndexType lhs_indextype_b = tmpl::at_c<LhsIndexTypeList, 1>::value;

#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_2_IMPL(_, data)                               \
  test_evaluate_rank_2_impl<                                                  \
      ReturnLhsTensor, DataType, RhsSymmetry,                                 \
      index_list<                                                             \
          ::Tensor_detail::TensorIndexType<DIM_A(data), TensorIndexA.valence, \
                                           Frame, rhs_indextype_a>,           \
          ::Tensor_detail::TensorIndexType<DIM_B(data), TensorIndexB.valence, \
                                           Frame, rhs_indextype_b>>,          \
      TensorIndexA, TensorIndexB, LhsSymmetry,                                \
      index_list<                                                             \
          ::Tensor_detail::TensorIndexType<DIM_A(data), TensorIndexA.valence, \
                                           Frame, lhs_indextype_a>,           \
          ::Tensor_detail::TensorIndexType<DIM_B(data), TensorIndexB.valence, \
                                           Frame, lhs_indextype_b>>>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_2_IMPL, (1, 2, 3), (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_2_IMPL

#undef DIM_B
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsIndexTypeList, auto& TensorIndexA, auto& TensorIndexB,
          typename Frame, typename LhsSymmetry = RhsSymmetry,
          typename LhsIndexTypeList = RhsIndexTypeList,
          Requires<std::is_same_v<RhsSymmetry, Symmetry<1, 1>>> = nullptr>
void test_evaluate_rank_2() {
  constexpr IndexType rhs_indextype_a = tmpl::at_c<RhsIndexTypeList, 0>::value;
  constexpr IndexType rhs_indextype_b = tmpl::at_c<RhsIndexTypeList, 1>::value;
  constexpr IndexType lhs_indextype_a = tmpl::at_c<LhsIndexTypeList, 0>::value;
  constexpr IndexType lhs_indextype_b = tmpl::at_c<LhsIndexTypeList, 1>::value;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define CALL_TEST_EVALUATE_RANK_2_IMPL(_, data)                             \
  test_evaluate_rank_2_impl<                                                \
      ReturnLhsTensor, DataType, RhsSymmetry,                               \
      index_list<                                                           \
          ::Tensor_detail::TensorIndexType<DIM(data), TensorIndexA.valence, \
                                           Frame, rhs_indextype_a>,         \
          ::Tensor_detail::TensorIndexType<DIM(data), TensorIndexB.valence, \
                                           Frame, rhs_indextype_b>>,        \
      TensorIndexA, TensorIndexB, LhsSymmetry,                              \
      index_list<                                                           \
          ::Tensor_detail::TensorIndexType<DIM(data), TensorIndexA.valence, \
                                           Frame, lhs_indextype_a>,         \
          ::Tensor_detail::TensorIndexType<DIM(data), TensorIndexB.valence, \
                                           Frame, lhs_indextype_b>>>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_2_IMPL, (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_2_IMPL

#undef DIM
}

}  // namespace TestHelpers::tenex
