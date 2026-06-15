// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/QuarticPolynomial.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace ScalarTensor::sgb::CouplingFunctions {

// Definitions for the QuarticPolynomial class
QuarticPolynomial::QuarticPolynomial(double linear, double quadratic,
                                     double cubic, double quartic)
    : linear_(linear), quadratic_(quadratic), cubic_(cubic),
    quartic_(quartic) {}

void QuarticPolynomial::coupling_function_impl(
    const gsl::not_null<Scalar<DataVector>*> function_values,
    const Scalar<DataVector>& scalar_field) const {
  get(*function_values) =
      get(scalar_field) *
      (linear_ +
       get(scalar_field) *
           (quadratic_ +
            get(scalar_field) * (cubic_ + get(scalar_field) * quartic_)));
}

void QuarticPolynomial::coupling_function_prime_impl(
    const gsl::not_null<Scalar<DataVector>*> function_values,
    const Scalar<DataVector>& scalar_field) const {
  get(*function_values) =
      linear_ + get(scalar_field) *
                    (2. * quadratic_ +
                     get(scalar_field) *
                         (3. * cubic_ + 4. * quartic_ * get(scalar_field)));
}

void QuarticPolynomial::coupling_function_prime_prime_impl(
    const gsl::not_null<Scalar<DataVector>*> function_values,
    const Scalar<DataVector>& scalar_field) const {
  get(*function_values) =
      2.0 * quadratic_ +
      get(scalar_field) * (6.0 * cubic_ + 12.0 * quartic_ * get(scalar_field));
}

void QuarticPolynomial::pup(PUP::er& p) {
  CouplingFunction::pup(p);
  p | linear_;
  p | quadratic_;
  p | cubic_;
  p | quartic_;
}
PUP::able::PUP_ID QuarticPolynomial::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarTensor::sgb::CouplingFunctions
