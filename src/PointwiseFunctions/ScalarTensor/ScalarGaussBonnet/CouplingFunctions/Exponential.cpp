// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/Exponential.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace ScalarTensor::sgb::CouplingFunctions {

Exponential::Exponential(double lambda, double gamma)
    : lambda_(lambda), gamma_(gamma) {}

void Exponential::coupling_function_impl(
    const gsl::not_null<Scalar<DataVector>*> function_values,
    const Scalar<DataVector>& scalar_field) const {
  get(*function_values) = lambda_ * exp(-gamma_ * get(scalar_field));
}

void Exponential::coupling_function_prime_impl(
    const gsl::not_null<Scalar<DataVector>*> function_values,
    const Scalar<DataVector>& scalar_field) const {
  get(*function_values) = -gamma_ * lambda_ * exp(-gamma_ * get(scalar_field));
}

void Exponential::coupling_function_prime_prime_impl(
    const gsl::not_null<Scalar<DataVector>*> function_values,
    const Scalar<DataVector>& scalar_field) const {
  get(*function_values) =
      gamma_ * gamma_ * lambda_ * exp(-gamma_ * get(scalar_field));
}

void Exponential::pup(PUP::er& p) {
  CouplingFunction::pup(p);
  p | lambda_;
  p | gamma_;
}

PUP::able::PUP_ID Exponential::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarTensor::sgb::CouplingFunctions
