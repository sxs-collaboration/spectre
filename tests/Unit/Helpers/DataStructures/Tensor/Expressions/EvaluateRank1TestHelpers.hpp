// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::TensorExpressions {

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 1 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam TensorIndexTypeList the Tensors' typelist containing their
/// \ref SpacetimeIndex "TensorIndexType"
/// \tparam TensorIndex the TensorIndex used in the the TensorExpression,
/// e.g. `ti_a`
template <typename DataType, typename TensorIndexTypeList, auto& TensorIndex>
void test_evaluate_rank_1_impl() noexcept {
  const size_t used_for_size = 5;
  using tensor_type = Tensor<DataType, Symmetry<1>, TensorIndexTypeList>;
  tensor_type R_a(used_for_size);
  std::iota(R_a.begin(), R_a.end(), 0.0);

  // L_a = R_a
  // Use explicit type (vs auto) so the compiler checks return type of
  // `evaluate`
  const tensor_type L_a_returned =
      ::TensorExpressions::evaluate<TensorIndex>(R_a(TensorIndex));
  tensor_type L_a_filled{};
  ::TensorExpressions::evaluate<TensorIndex>(make_not_null(&L_a_filled),
                                             R_a(TensorIndex));

  const size_t dim = tmpl::at_c<TensorIndexTypeList, 0>::dim;

  // For L_a = R_a, check that L_i == R_i
  for (size_t i = 0; i < dim; ++i) {
    CHECK(L_a_returned.get(i) == R_a.get(i));
    CHECK(L_a_filled.get(i) == R_a.get(i));
  }

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    // L_a = R_a
    Variables<tmpl::list<::Tags::TempTensor<1, tensor_type>>> L_a_var{
        used_for_size};
    tensor_type& L_a_temp = get<::Tags::TempTensor<1, tensor_type>>(L_a_var);
    ::TensorExpressions::evaluate<TensorIndex>(make_not_null(&L_a_temp),
                                               R_a(TensorIndex));

    // For L_a = R_a, check that L_i == R_i
    for (size_t i = 0; i < dim; ++i) {
      CHECK(L_a_temp.get(i) == R_a.get(i));
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluating single rank 1 Tensors on multiple Frame
/// types and dimensions
///
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam TensorIndexType the Tensors' \ref SpacetimeIndex "TensorIndexType"
/// \tparam Valence the valence of the Tensors' index
/// \tparam TensorIndex the TensorIndex used in the the TensorExpression,
/// e.g. `ti_a`
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexType, UpLo Valence,
          auto& TensorIndex>
void test_evaluate_rank_1() noexcept {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_1_IMPL(_, data)                               \
  test_evaluate_rank_1_impl<                                                  \
      DataType, index_list<TensorIndexType<DIM(data), Valence, FRAME(data)>>, \
      TensorIndex>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_1_IMPL, (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_1_IMPL
#undef FRAME
#undef DIM
}

}  // namespace TestHelpers::TensorExpressions
