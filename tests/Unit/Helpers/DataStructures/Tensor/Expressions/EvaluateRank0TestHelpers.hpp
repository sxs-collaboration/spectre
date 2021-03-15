// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"

namespace TestHelpers::TensorExpressions {

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 0 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \param data the data being stored in the Tensors
template <typename DataType>
void test_evaluate_rank_0(const DataType& data) noexcept {
  const Tensor<DataType> R{{{data}}};

  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  const Tensor<DataType> L_returned = ::TensorExpressions::evaluate(R());
  Tensor<DataType> L_filled{};
  ::TensorExpressions::evaluate(make_not_null(&L_filled), R());

  CHECK(L_returned.get() == data);
  CHECK(L_filled.get() == data);

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    Variables<tmpl::list<::Tags::TempTensor<1, Tensor<DataType>>>> L_var{
        data.size()};
    Tensor<DataType>& L_temp =
        get<::Tags::TempTensor<1, Tensor<DataType>>>(L_var);
    ::TensorExpressions::evaluate(make_not_null(&L_temp), R());

    CHECK(L_temp.get() == data);
  }
}

}  // namespace TestHelpers::TensorExpressions
