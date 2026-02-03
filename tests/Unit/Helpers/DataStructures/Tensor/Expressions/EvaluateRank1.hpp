// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
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
#include "Utilities/TMPL.hpp"

namespace TestHelpers::tenex {

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 1 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details See `test_evaluate_rank_3_impl` for general details.
///
/// \tparam ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not, which instead tests evaluation by
/// assigning to the result tensor passed in as an argument
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam TensorIndex the TensorIndex used in the the TensorExpression,
/// e.g. `ti::a`
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
template <bool ReturnLhsTensor, typename DataType,
          typename RhsTensorIndexTypeList, auto& TensorIndex,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_1_impl() {
  using symmetry = Symmetry<1>;
  using L_a_type = Tensor<DataType, symmetry, LhsTensorIndexTypeList>;
  using R_a_type = Tensor<DataType, symmetry, RhsTensorIndexTypeList>;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-5.0, 5.0);
  const size_t used_for_size = 3;
  const auto R_a = make_with_random_values<R_a_type>(
      make_not_null(&generator), distribution, used_for_size);
  auto expected_L_a =
      ReturnLhsTensor
          ? L_a_type{}
          : make_with_value<L_a_type>(
                used_for_size, component_placeholder_value<DataType>::value);

  using lhs_tensorindextype = tmpl::at_c<LhsTensorIndexTypeList, 0>;
  using rhs_tensorindextype = tmpl::at_c<RhsTensorIndexTypeList, 0>;

  const std::pair<size_t, size_t> lhs_index_value_range =
      get_index_value_range<lhs_tensorindextype, TensorIndex>();
  const std::pair<size_t, size_t> rhs_index_value_range =
      get_index_value_range<rhs_tensorindextype, TensorIndex>();

  for (size_t lhs_a = lhs_index_value_range.first,
              rhs_a = rhs_index_value_range.first;
       lhs_a <= lhs_index_value_range.second; lhs_a++, rhs_a++) {
    expected_L_a.get(lhs_a) = R_a.get(rhs_a);
  }

  // L_a = R_a
  L_a_type L_a(used_for_size);
  // component placeholder is used to detect which components have incorrectly
  // or correctly (in the case of using spatial or time indices for spacetime
  // indices) not been modified by evaluation of the RHS expression
  std::fill(L_a.begin(), L_a.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndex>(make_not_null(&L_a),
                                              R_a(TensorIndex));

  CHECK(L_a == expected_L_a);  // check LHS evaluated correctly

  // Test with Variables
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<::Tags::TempTensor<0, R_a_type>,
                         ::Tags::TempTensor<1, L_a_type>>>
        vars(used_for_size, std::numeric_limits<double>::signaling_NaN());

    R_a_type& R_a_temp = get<::Tags::TempTensor<0, R_a_type>>(vars);
    R_a_temp = R_a;

    // L_a = R_a
    L_a_type& L_a_temp = get<::Tags::TempTensor<1, L_a_type>>(vars);
    std::fill(L_a_temp.begin(), L_a_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndex>(make_not_null(&L_a_temp),
                                                R_a(TensorIndex));

    CHECK(R_a_temp == R_a);           // check RHS wasn't modified
    CHECK(L_a_temp == expected_L_a);  // check LHS evaluated correctly
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluating single rank 1 Tensors on multiple
/// dimensions
///
/// \details See `test_evaluate_rank_3_impl` for general details.
///
/// \tparam ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not, which instead tests evaluation by
/// assigning to the result tensor passed in as an argument
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsIndexTypeList the RHS Tensor's integral list of `IndexType`s
/// \tparam TensorIndex the TensorIndex used in the the TensorExpression,
/// e.g. `ti::a`
/// \tparam Frame the frame of the tensor index
/// \tparam LhsIndexTypeList the LHS Tensor's integral list of `IndexType`s
template <bool ReturnLhsTensor, typename DataType, typename RhsIndexTypeList,
          auto& TensorIndex, typename Frame,
          typename LhsIndexTypeList = RhsIndexTypeList>
void test_evaluate_rank_1() {
  constexpr IndexType rhs_indextype = tmpl::at_c<RhsIndexTypeList, 0>::value;
  constexpr IndexType lhs_indextype = tmpl::at_c<LhsIndexTypeList, 0>::value;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define CALL_TEST_EVALUATE_RANK_1_IMPL(_, data)                   \
  test_evaluate_rank_1_impl<                                      \
      ReturnLhsTensor, DataType,                                  \
      index_list<::Tensor_detail::TensorIndexType<                \
          DIM(data), TensorIndex.valence, Frame, rhs_indextype>>, \
      TensorIndex,                                                \
      index_list<::Tensor_detail::TensorIndexType<                \
          DIM(data), TensorIndex.valence, Frame, lhs_indextype>>>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_1_IMPL, (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_1_IMPL

#undef DIM
}

}  // namespace TestHelpers::tenex
