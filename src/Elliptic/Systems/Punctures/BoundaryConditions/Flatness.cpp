// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Punctures/BoundaryConditions/Flatness.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace Punctures::BoundaryConditions {

void Flatness::apply(
    const gsl::not_null<Scalar<DataVector>*> field,
    const gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient,
    const tnsr::I<DataVector, 3>& x) {
  get(*n_dot_field_gradient) = -get(*field) / get(magnitude(x));
}

void Flatness::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> field_correction,
    const gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient_correction,
    const tnsr::I<DataVector, 3>& x) {
  get(*n_dot_field_gradient_correction) =
      -get(*field_correction) / get(magnitude(x));
}

bool operator==(const Flatness& /*lhs*/, const Flatness& /*rhs*/) {
  return true;
}

bool operator!=(const Flatness& lhs, const Flatness& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID Flatness::my_PUP_ID = 0;  // NOLINT

}  // namespace Punctures::BoundaryConditions
