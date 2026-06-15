// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/CouplingFunction.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace ScalarTensor::sgb::CouplingFunctions {
void CouplingFunction::coupling_function(
    const gsl::not_null<Scalar<DataVector>*> function_values,
    const Scalar<DataVector>& scalar_field) const {
  set_number_of_grid_points(function_values, scalar_field);
  coupling_function_impl(function_values, scalar_field);
}

Scalar<DataVector> CouplingFunction::coupling_function(
    const Scalar<DataVector>& scalar_field) const {
  Scalar<DataVector> result =
      make_with_value<Scalar<DataVector>>(get(scalar_field), 0.0);
  coupling_function_impl(make_not_null(&result), scalar_field);
  return result;
}

void CouplingFunction::coupling_function_prime(
    const gsl::not_null<Scalar<DataVector>*> function_values,
    const Scalar<DataVector>& scalar_field) const {
  set_number_of_grid_points(function_values, scalar_field);
  coupling_function_prime_impl(function_values, scalar_field);
}

Scalar<DataVector> CouplingFunction::coupling_function_prime(
    const Scalar<DataVector>& scalar_field) const {
  Scalar<DataVector> result =
      make_with_value<Scalar<DataVector>>(get(scalar_field), 0.0);
  coupling_function_prime_impl(make_not_null(&result), scalar_field);
  return result;
}

void CouplingFunction::coupling_function_prime_prime(
    const gsl::not_null<Scalar<DataVector>*> function_values,
    const Scalar<DataVector>& scalar_field) const {
  set_number_of_grid_points(function_values, scalar_field);
  coupling_function_prime_prime_impl(function_values, scalar_field);
}

Scalar<DataVector> CouplingFunction::coupling_function_prime_prime(
    const Scalar<DataVector>& scalar_field) const {
  Scalar<DataVector> result =
      make_with_value<Scalar<DataVector>>(get(scalar_field), 0.0);
  coupling_function_prime_prime_impl(make_not_null(&result), scalar_field);
  return result;
}
}  // namespace ScalarTensor::sgb::CouplingFunctions
