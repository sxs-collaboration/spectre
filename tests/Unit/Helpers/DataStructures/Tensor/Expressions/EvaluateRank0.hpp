// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <random>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComponentPlaceholder.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::tenex {

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 0 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \tparam DataType the type of data being stored in the Tensors
template <typename DataType>
void test_evaluate_rank_0() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-5.0, 5.0);
  const size_t used_for_size = 3;
  const auto R = make_with_random_values<Tensor<DataType>>(
      make_not_null(&generator), distribution, used_for_size);

  Scalar<DataType> L(used_for_size);
  // component placeholder is used to detect if LHS scalar was not modified
  std::fill(L.begin(), L.end(), component_placeholder_value<DataType>::value);
  call_evaluate<true>(make_not_null(&L), R());

  CHECK(L == R);  // check LHS evaluated correctly

  // Test with Variables
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<::Tags::TempTensor<0, Scalar<DataType>>,
                         ::Tags::TempTensor<1, Scalar<DataType>>>>
        vars(used_for_size, std::numeric_limits<double>::signaling_NaN());

    Scalar<DataType>& R_temp =
        get<::Tags::TempTensor<0, Scalar<DataType>>>(vars);
    get(R_temp) = get(R);

    Scalar<DataType>& L_temp =
        get<::Tags::TempTensor<1, Scalar<DataType>>>(vars);
    call_evaluate<true>(make_not_null(&L_temp), R());

    CHECK(R_temp == R);  // check RHS wasn't modified
    CHECK(L_temp == R);  // check LHS evaluated correctly
  }
}

}  // namespace TestHelpers::tenex
