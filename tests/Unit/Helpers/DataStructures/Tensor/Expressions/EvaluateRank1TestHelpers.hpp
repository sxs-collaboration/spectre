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
/// \param tensorindex the TensorIndex used in the the TensorExpression,
/// e.g. `ti_a`
template <typename DataType, typename TensorIndexTypeList, typename TensorIndex>
void test_evaluate_rank_1_impl(const TensorIndex& tensorindex) noexcept {
  Tensor<DataType, Symmetry<1>, TensorIndexTypeList> R_a(5_st);
  std::iota(R_a.begin(), R_a.end(), 0.0);

  // L_a = R_a
  // Use explicit type (vs auto) so the compiler checks return type of
  // `evaluate`
  const Tensor<DataType, Symmetry<1>, TensorIndexTypeList> L_a =
      ::TensorExpressions::evaluate<TensorIndex>(R_a(tensorindex));

  const size_t dim = tmpl::at_c<TensorIndexTypeList, 0>::dim;

  // For L_a = R_a, check that L_i == R_i
  for (size_t i = 0; i < dim; ++i) {
    CHECK(L_a.get(i) == R_a.get(i));
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluating single rank 1 Tensors on multiple Frame
/// types and dimensions
///
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam TensorIndexType the Tensors' \ref SpacetimeIndex "TensorIndexType"
/// \tparam Valence the valence of the Tensors' index
/// \param tensorindex the TensorIndex used in the the TensorExpression,
/// e.g. `ti_a`
template <typename DataType,
          template <size_t, UpLo, typename> class TensorIndexType, UpLo Valence,
          typename TensorIndex>
void test_evaluate_rank_1(const TensorIndex& tensorindex) noexcept {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_1_IMPL(_, data)                                \
  test_evaluate_rank_1_impl<                                                   \
      DataType, index_list<TensorIndexType<DIM(data), Valence, FRAME(data)>>>( \
      tensorindex);

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_1_IMPL, (1, 2, 3),
                          (Frame::Grid, Frame::Inertial))

#undef CALL_TEST_EVALUATE_RANK_1_IMPL
#undef FRAME
#undef DIM
}

}  // namespace TestHelpers::TensorExpressions
