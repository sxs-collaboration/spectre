// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/None.hpp"

namespace ScalarSelfForce::BoundaryConditions {

None::None(CkMigrateMessage* m) : Base(m) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> None::get_clone()
    const {
  return std::make_unique<None>(*this);
}

void None::apply(
    const gsl::not_null<Scalar<ComplexDataVector>*> /*field*/,
    const gsl::not_null<Scalar<ComplexDataVector>*> /*n_dot_field_gradient*/,
    const tnsr::i<ComplexDataVector, 2>& /*deriv_field*/) const {
  // Nothing to do
}

void None::apply_linearized(
    const gsl::not_null<Scalar<ComplexDataVector>*> /*field_correction*/,
    const gsl::not_null<Scalar<ComplexDataVector>*>
    /*n_dot_field_gradient_correction*/,
    const tnsr::i<ComplexDataVector, 2>& /*deriv_field_correction*/) const {
  // Nothing to do
}

bool operator==(const None& /*lhs*/, const None& /*rhs*/) { return true; }

bool operator!=(const None& lhs, const None& rhs) { return not(lhs == rhs); }

PUP::able::PUP_ID None::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarSelfForce::BoundaryConditions
